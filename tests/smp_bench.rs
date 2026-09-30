use chess_engine::engine::{self, ChessEngine, Limits, Score, SearchStats, TimeMode};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use std::time::{Duration, Instant};

/// Run explicitly with:
///   cargo test --release --test smp_bench -- --ignored --nocapture
///
/// Parallel scaling plus the YBWC tuning counters, at a fixed depth so every
/// thread count does the same *search*, not the same amount of time.
///
/// Read it in release only. In debug the atomics, the iterator adapters and the
/// 2.4 KiB `OrderedMoves` moves are all real function calls and memcpys, which
/// penalises exactly the code the parallel search added.
///
/// The correctness check is deliberately a tolerance rather than equality.
///
/// The search is not exact: `history` is shared and mutated concurrently, and it
/// feeds `move_order_score`, which sets `move_count`, which sets
/// `lmr_reduction`. Change the order in which siblings finish — which is all
/// parallelism does — and reductions land differently, transposition entries
/// become visible to other subtrees at different moments, and null-move
/// verifications come out differently. Small score drift is therefore expected
/// and is not a bug; every parallel engine has it.
///
/// What would be a bug is drift that is large, that flips a mate score, or that
/// appears at shallow depths where no split points fire. So: mate scores must
/// match exactly, and centipawn scores must agree within `MAX_SCORE_DRIFT_CP`.
/// A race in the merge or the cancel chain produces far worse than that.
///
/// What to look at:
///   idle%    the headline. Under ~10% on four threads is healthy; above ~30%
///            the bottleneck is how long an idle worker waits for somebody to
///            reach a split decision, which SPLIT_DEPTH tuning only partly fixes.
///   splits   vs `refused`: few splits with high idle% means SPLIT_DEPTH is too
///            high. Many refusals with low idle% just means you are saturated.
///   wasted   nodes searched after the split point was cancelled. If this stays
///            small while the node count grows with threads, the extra work is
///            coming from stale alpha, not from speculation past a cutoff.
///   dem.cut  of `demoted` (CUT nodes that split after CUT_DEMOTE), how many cut
///            anyway. A high ratio means CUT_DEMOTE is too low.
const DEPTH: u8 = 13;
const HASH_MB: usize = 512;
/// Tclerated score drift against the single-threaded search. Generous enough for
/// ordinary instability, far tighter than anything a real race would cause.
const MAX_SCORE_DRIFT_CP: i32 = 50;

const POSITIONS: [(&str, &str); 3] = [
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ),
    (
        "tactical",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    ),
];

#[derive(Default)]
struct Totals {
    elapsed: Duration,
    nodes: u64,
    splits: u64,
    refused: u64,
    demoted: u64,
    demoted_cut: u64,
    wasted: u64,
    helper_nanos: u64,
}

impl Totals {
    fn add(&mut self, elapsed: Duration, nodes: u64, s: &SearchStats) {
        self.elapsed += elapsed;
        self.nodes += nodes;
        self.splits += s.splits;
        self.refused += s.splits_refused_busy;
        self.demoted += s.demoted_splits;
        self.demoted_cut += s.demoted_splits_cut;
        self.wasted += s.wasted_nodes;
        self.helper_nanos += s.helper_nanos;
    }

    /// Same formula as `SearchStats::idle_fraction`, over the summed searches.
    fn idle_fraction(&self, threads: usize) -> f64 {
        let wall = self.elapsed.as_nanos() as f64;
        let available = wall * threads as f64;
        if available <= 0.0 {
            return 0.0;
        }
        (1.0 - (wall + self.helper_nanos as f64) / available).max(0.0)
    }
}

/// `Score` has `Display` but not `Debug`, and these are `Option<Score>`.
fn show(s: &Option<Score>) -> String {
    s.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "none".into())
}

/// `None` when the two are not even the same kind of score — always a failure.
fn drift(a: &Option<Score>, b: &Option<Score>) -> Option<i32> {
    match (a, b) {
        (None, None) => Some(0),
        (Some(Score::Cp(x)), Some(Score::Cp(y))) => Some((x - y).abs()),
        (Some(Score::Mate(x)), Some(Score::Mate(y))) => {
            if x == y { Some(0) } else { None }
        }
        _ => None,
    }
}

fn position(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .expect("valid fen")
        .into_position(CastlingMode::Standard)
        .expect("legal position")
}

/// 1, 2, 4, … up to what the machine actually offers, always including 1.
fn thread_counts() -> Vec<usize> {
    let max = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut counts: Vec<usize> = [1usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&t| t <= max)
        .collect();
    if counts.is_empty() {
        counts.push(1);
    }
    counts
}

#[test]
#[ignore = "benchmark; run with --release --ignored --nocapture"]
fn smp_scaling() {
    println!(
        "depth {DEPTH}, {} positions, {HASH_MB} MiB hash",
        POSITIONS.len()
    );
    println!(
        "{:>3} {:>9} {:>11} {:>7} {:>8} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "thr", "time", "nodes", "idle%", "splits", "refused", "demoted", "wasted", "dem.cut", "speedup"
    );

    let mut max_drift = 0i32;
    let mut baseline_scores: Option<Vec<Option<Score>>> = None;
    let mut baseline_time: Option<Duration> = None;

    for threads in thread_counts() {
        let mut names: Vec<&str> = Vec::new();
        let mut totals = Totals::default();
        let mut scores: Vec<Option<Score>> = Vec::new();

        for (name, fen) in POSITIONS {
            let pos = position(fen);
            let mut engine = engine::SearchEngine::new(HASH_MB, threads);
            engine.search_handle().set_limits(Limits {
                time_mode: TimeMode::Unbound,
                max_depth: Some(DEPTH),
                max_nodes: None,
                restrict_to: None,
            });
            engine.set_position(pos, &[]);

            let started = Instant::now();
            let result = engine.search(&mut |_| {});
            let elapsed = started.elapsed();

            totals.add(elapsed, result.nodes, &engine.search_stats());
            names.push(name);
            scores.push(result.score);
        }

        let speedup = baseline_time.map_or(1.0, |b| b.as_secs_f64() / totals.elapsed.as_secs_f64());
        println!(
            "{:>3} {:>8.3}s {:>11} {:>6.1}% {:>8} {:>8} {:>8} {:>10} {:>8} {:>7.2}x",
            threads,
            totals.elapsed.as_secs_f64(),
            totals.nodes,
            totals.idle_fraction(threads) * 100.0,
            totals.splits,
            totals.refused,
            totals.demoted,
            totals.wasted,
            totals.demoted_cut,
            speedup,
        );

        match &baseline_scores {
            None => {
                baseline_scores = Some(scores);
                baseline_time = Some(totals.elapsed);
            }
            Some(expected) => {
                for ((name, got), want) in names.iter().zip(&scores).zip(expected) {
                    let d = drift(got, want).unwrap_or_else(|| {
                        panic!(
                            "{name} at {threads} threads: {} is not comparable to the \
                             single-threaded {}",
                            show(got), show(want)
                        )
                    });
                    assert!(
                        d <= MAX_SCORE_DRIFT_CP,
                        "{name} at {threads} threads drifted {d} cp from the single-threaded \
                         search ({} vs {}); tolerance is {MAX_SCORE_DRIFT_CP}",
                        show(got), show(want)
                    );
                    max_drift = max_drift.max(d);
                }
            }
        }
    }

    println!("max score drift vs 1 thread: {max_drift} cp (tolerance {MAX_SCORE_DRIFT_CP})");
}
