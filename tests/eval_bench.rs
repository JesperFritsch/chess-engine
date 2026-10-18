use chess_engine::engine::{self, ChessEngine, Evaluator, MaterialEval, PstEval, SearchResult};
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::{CastlingMode, Chess};
use std::time::{Duration, Instant};

/// Compares evaluators on speed and on what they make the search do.
///
/// Run explicitly with:
///   cargo test --release --test eval_bench -- --ignored --nocapture
/// Search depth defaults to 10; override with `BENCH_DEPTH=12`. Set
/// `NNUE_NET=path/to/net.bin` to include a trained net.
///
/// Every evaluator searches the same positions to the same fixed depth, on one
/// thread, with a fresh hash table per position, so the runs are independent
/// and repeatable. Node counts are then deterministic; only the time varies.
#[test]
#[ignore = "manual benchmark; run with --ignored --nocapture"]
fn bench_evaluators() {
    let depth: u8 = std::env::var("BENCH_DEPTH")
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(10);
    let positions: Vec<(&str, Chess)> = POSITIONS.iter().map(|(name, fen)| (*name, parse(fen))).collect();

    // Add new evaluators here.
    let mut runs = vec![
        run("pst", PstEval, &positions, depth),
        run("material", MaterialEval, &positions, depth),
    ];
    // NNUE_NET=nets/net.bin adds the net to the comparison.
    if let Ok(path) = std::env::var("NNUE_NET") {
        runs.push(run("nnue", engine::nnue::load_evaluator(&path).unwrap(), &positions, depth));
    }

    println!("\ndepth {depth}, {} positions, 1 thread\n", positions.len());
    println!("{:<10} {:>14} {:>10} {:>10}", "evaluator", "total nodes", "time", "knps");
    for r in &runs {
        let nodes: u64 = r.results.iter().map(|x| x.nodes).sum();
        let secs = r.time.as_secs_f64();
        println!("{:<10} {:>14} {:>9.3}s {:>10.0}", r.name, nodes, secs, nodes as f64 / secs / 1000.0);
    }

    // Per position: what each evaluator chose, and how much it searched to get there.
    println!();
    print!("{:<12}", "position");
    for r in &runs {
        print!(" | {:<28}", r.name);
    }
    println!();
    for (i, (name, pos)) in positions.iter().enumerate() {
        print!("{:<12}", name);
        for r in &runs {
            let res = &r.results[i];
            let mv = res.best_move.map_or("-".to_string(), |m| San::from_move(pos, m).to_string());
            let score = res.score.as_ref().map_or("-".to_string(), |s| s.to_string());
            print!(" | {:<7} {:>8} {:>11}", mv, score, res.nodes);
        }
        println!();
    }
}

struct Run {
    name: &'static str,
    time: Duration,
    results: Vec<SearchResult>,
}

fn run<E: Evaluator>(name: &'static str, eval: E, positions: &[(&str, Chess)], depth: u8) -> Run {
    let mut time = Duration::ZERO;
    let mut results = Vec::new();
    for (_, pos) in positions {
        let mut engine = engine::SearchEngine::with_evaluator(eval.clone(), 16, 1);
        engine.search_handle().set_limits(engine::Limits {
            time_mode: engine::TimeMode::Unbound,
            max_depth: Some(depth),
            max_nodes: None,
            restrict_to: None,
        });
        engine.set_position(pos.clone(), &[]);
        let t = Instant::now();
        results.push(engine.search(&mut |_| {}));
        time += t.elapsed();
    }
    Run { name, time, results }
}

fn parse(fen: &str) -> Chess {
    fen.parse::<Fen>()
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap_or_else(|e| panic!("illegal bench position {fen}: {e}"))
}

/// Opening, middlegame and endgame positions, so an evaluator that is only good
/// (or only fast) in one phase shows up.
const POSITIONS: &[(&str, &str)] = &[
    ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("perft4",     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"),
    ("perft5",     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
    ("italian",    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("middle1",    "r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14"),
    ("middle2",    "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24"),
    ("closed",     "r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42"),
    ("rook-end",   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("pawn-end",   "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"),
    ("kpk",        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"),
    ("krk",        "8/8/4k3/8/8/8/3R4/3K4 w - - 0 1"),
];
