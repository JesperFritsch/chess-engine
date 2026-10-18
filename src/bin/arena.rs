//! Evaluator-vs-evaluator match.
//!
//!   cargo run --release --bin arena -- pst material --games 200 --nodes 25000
//!
//! Every opening is played twice with colors swapped, so an unbalanced random
//! opening helps both sides equally. Results are from the first evaluator's
//! point of view.

use chess_engine::arena::{self, make_engine, GameConfig, GameResult, Job, Rng, Termination, EVALUATORS};
use chess_engine::engine::{Limits, TimeMode};
use std::collections::BTreeMap;
use std::process::exit;

struct Args {
    a: String,
    b: String,
    games: usize,
    nodes: u64,
    concurrency: usize,
    opening_plies: usize,
    seed: u64,
    hash_mb: usize,
    json: Option<String>,
}

fn main() {
    let args = parse_args();
    let config = GameConfig {
        limits: Limits {
            time_mode: TimeMode::Unbound,
            max_depth: None,
            max_nodes: Some(args.nodes),
            restrict_to: None,
        },
        adjudication: Default::default(),
    };

    // Openings in pairs: job 2k and 2k+1 share an opening, colors swapped.
    let mut rng = Rng::new(args.seed);
    let jobs: Vec<Job> = (0..args.games / 2)
        .flat_map(|_| {
            let seed = rng.next_u64();
            [Job { seed, first_is_white: true }, Job { seed, first_is_white: false }]
        })
        .collect();

    println!(
        "{} vs {}: {} games, {} nodes/move, {} threads, {}-ply random openings, seed {}",
        args.a, args.b, jobs.len(), args.nodes, args.concurrency, args.opening_plies, args.seed
    );

    // Score of the first evaluator per game, indexed like `jobs`.
    let mut scores: Vec<Option<f64>> = vec![None; jobs.len()];
    let mut terminations: BTreeMap<String, usize> = BTreeMap::new();
    let (mut w, mut d, mut l) = (0, 0, 0);

    arena::run_games(
        &jobs,
        args.concurrency,
        &config,
        || (make_engine(&args.a, args.hash_mb).unwrap(), make_engine(&args.b, args.hash_mb).unwrap()),
        // Purely random: an opening built with one side's engine would differ
        // between the two games of a pair whenever the engines differ.
        |rng, _| arena::random_opening(rng, args.opening_plies),
        |i, job, record| {
            let a_color = if job.first_is_white { shakmaty::Color::White } else { shakmaty::Color::Black };
            let s = record.result.score_for(a_color);
            scores[i] = Some(s);
            match s {
                1.0 => w += 1,
                0.5 => d += 1,
                _ => l += 1,
            }
            *terminations.entry(termination_name(record.termination, record.result)).or_default() += 1;
            let n = w + d + l;
            let (elo, margin) = elo(&pair_scores(&scores));
            print!("\r{n}/{}  W {w}  D {d}  L {l}   elo {elo:+.0} ± {margin:.0}   ", jobs.len());
            use std::io::Write;
            std::io::stdout().flush().ok();
        },
    );

    let n = w + d + l;
    let (elo, margin) = elo(&pair_scores(&scores));
    println!("\n");
    println!("{} vs {}", args.a, args.b);
    println!("  games   {n}   W {w}  D {d}  L {l}   score {:.1}%", 100.0 * (w as f64 + 0.5 * d as f64) / n as f64);
    println!("  elo     {elo:+.1} ± {margin:.1} (95%)");
    println!("  endings:");
    for (name, count) in &terminations {
        println!("    {name:<24} {count}");
    }

    // A machine-readable summary, for scripts such as nnue/loop.py.
    if let Some(path) = &args.json {
        let json = format!(
            "{{\"a\": {:?}, \"b\": {:?}, \"games\": {n}, \"wins\": {w}, \"draws\": {d}, \"losses\": {l}, \"elo\": {elo:.2}, \"margin\": {}}}\n",
            args.a, args.b, if margin.is_finite() { format!("{margin:.2}") } else { "null".into() }
        );
        std::fs::write(path, json).unwrap_or_else(|e| {
            eprintln!("cannot write {path}: {e}");
            exit(1)
        });
    }
}

/// Scores of complete opening pairs (0, 0.5, 1, 1.5 or 2 per pair).
fn pair_scores(scores: &[Option<f64>]) -> Vec<f64> {
    scores
        .chunks(2)
        .filter_map(|p| match p {
            [Some(a), Some(b)] => Some(a + b),
            _ => None,
        })
        .collect()
}

/// Elo difference and its 95% margin, from opening-pair scores.
///
/// Treating each pair as one sample (the "pentanomial" model) is the honest
/// choice here: the two games of a pair share an opening, so they are not
/// independent, and counting them separately would understate the error.
fn elo(pairs: &[f64]) -> (f64, f64) {
    let n = pairs.len() as f64;
    if n < 2.0 {
        return (0.0, f64::INFINITY);
    }
    let mean = pairs.iter().sum::<f64>() / n / 2.0;
    let var = pairs.iter().map(|p| (p / 2.0 - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (var / n).sqrt();
    let to_elo = |s: f64| -400.0 * (1.0 / s.clamp(1e-6, 1.0 - 1e-6) - 1.0).log10();
    let e = to_elo(mean);
    let margin = (to_elo(mean + 1.96 * se) - to_elo(mean - 1.96 * se)) / 2.0;
    (e, margin)
}

fn termination_name(t: Termination, r: GameResult) -> String {
    let side = match r {
        GameResult::WhiteWins => " (white)",
        GameResult::BlackWins => " (black)",
        GameResult::Draw => "",
    };
    format!("{t:?}{side}")
}

fn parse_args() -> Args {
    let usage = || -> ! {
        eprintln!(
            "usage: arena <eval-a> <eval-b> [--games N] [--nodes N] [--concurrency N] \
             [--opening-plies N] [--seed N] [--hash MB] [--json FILE]\nevaluators: {}",
            EVALUATORS.join(", ")
        );
        exit(2)
    };
    let mut args = Args {
        a: String::new(),
        b: String::new(),
        games: 200,
        nodes: 25_000,
        concurrency: std::thread::available_parallelism().map_or(1, |n| n.get()),
        opening_plies: 8,
        seed: 1,
        hash_mb: 16,
        json: None,
    };
    let mut positional = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage());
        match arg.as_str() {
            "--games" => args.games = value(),
            "--nodes" => args.nodes = value() as u64,
            "--concurrency" => args.concurrency = value(),
            "--opening-plies" => args.opening_plies = value(),
            "--seed" => args.seed = value() as u64,
            "--hash" => args.hash_mb = value(),
            "--json" => args.json = Some(it.next().unwrap_or_else(|| usage())),
            "-h" | "--help" => usage(),
            _ if arg.starts_with("--") => usage(),
            _ => positional.push(arg),
        }
    }
    let [a, b] = positional.try_into().unwrap_or_else(|_| usage());
    // Fails early, with a message, on an unknown name or an unreadable net.
    for name in [&a, &b] {
        if let Err(e) = make_engine(name, 1) {
            eprintln!("{e}");
            exit(2)
        }
    }
    if args.games < 2 {
        usage();
    }
    args.a = a;
    args.b = b;
    args
}
