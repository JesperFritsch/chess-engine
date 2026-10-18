//! Self-play training data for the NNUE.
//!
//!   cargo run --release --bin datagen -- --eval pst --games 10000 --nodes 5000 --out data.txt
//!
//! The engine plays itself from openings where every `--random-every`-th ply
//! is random and the rest are the engine's, with `--random-plies` (or one more)
//! random plies, and openings scored beyond `--max-opening-score` are discarded
//! (see `arena::mixed_opening`).
//! Every searched position after the opening that passes the filters becomes
//! one line:
//!
//!   <fen> | <score> | <result>
//!
//! `score` is the search score in centipawns and `result` the game's outcome
//! (1.0 White won, 0.5 draw, 0.0 Black won), both from White's point of view.
//! The FEN says who is to move, so a trainer can flip to the side to move.
//!
//! Lines are appended, and the file is flushed after every game: stopping the
//! run loses only the games in progress, and several runs can share one file.

use chess_engine::arena::{self, make_engine, GameConfig, GameRecord, GameResult, Job, OpeningConfig, PlyRecord, Rng, EVALUATORS, MATE_CP};
use chess_engine::engine::{Limits, TimeMode};
use shakmaty::fen::Fen;
use shakmaty::{Color, EnPassantMode, Position};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::process::exit;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Args {
    eval: String,
    games: usize,
    nodes: u64,
    concurrency: usize,
    opening: OpeningConfig,
    seed: u64,
    hash_mb: usize,
    out: String,
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

    let mut rng = Rng::new(args.seed);
    let jobs: Vec<Job> = (0..args.games)
        .map(|_| Job { seed: rng.next_u64(), first_is_white: true })
        .collect();

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .unwrap_or_else(|e| {
            eprintln!("cannot open {}: {e}", args.out);
            exit(1)
        });
    let mut out = BufWriter::new(file);

    println!(
        "datagen: {} self-play, {} games, {} nodes/move, {} threads, openings of {}-{} random plies, one every {} plies, within ±{} cp, seed {}, appending to {}",
        args.eval, args.games, args.nodes, args.concurrency, args.opening.random_plies,
        args.opening.random_plies + 1, args.opening.random_every, args.opening.max_score, args.seed, args.out
    );

    let start = Instant::now();
    let (mut games, mut written, mut seen) = (0usize, 0usize, 0usize);
    // Games won by White, drawn, won by Black.
    let mut results = [0usize; 3];
    arena::run_games(
        &jobs,
        args.concurrency,
        &config,
        || (make_engine(&args.eval, args.hash_mb).unwrap(), make_engine(&args.eval, args.hash_mb).unwrap()),
        |rng, engine| arena::mixed_opening(rng, engine, &config.limits, &args.opening),
        |_, _, record| {
            seen += record.plies.len();
            written += write_game(&mut out, &record);
            results[match record.result {
                GameResult::WhiteWins => 0,
                GameResult::Draw => 1,
                GameResult::BlackWins => 2,
            }] += 1;
            out.flush().unwrap_or_else(|e| {
                eprintln!("\nwrite failed: {e}");
                exit(1)
            });
            games += 1;
            let secs = start.elapsed().as_secs_f64();
            print!(
                "\r{games}/{}  positions {written} ({:.0}% kept)  {:.0} pos/s   ",
                args.games,
                100.0 * written as f64 / seen.max(1) as f64,
                written as f64 / secs
            );
            std::io::stdout().flush().ok();
        },
    );
    let secs = start.elapsed().as_secs_f64();
    println!("\ndone: {written} positions from {games} games in {secs:.1}s");

    // A machine-readable summary, for scripts such as nnue/loop.py.
    if let Some(path) = &args.json {
        let json = format!(
            "{{\"eval\": {:?}, \"games\": {games}, \"positions\": {written}, \"searched\": {seen}, \"seconds\": {secs:.1}, \"white_wins\": {}, \"draws\": {}, \"black_wins\": {}}}\n",
            args.eval, results[0], results[1], results[2]
        );
        std::fs::write(path, json).unwrap_or_else(|e| {
            eprintln!("cannot write {path}: {e}");
            exit(1)
        });
    }
}

/// Write the game's usable positions; returns how many.
fn write_game(out: &mut impl Write, record: &GameRecord) -> usize {
    let result = record.result.score_for(Color::White);
    let mut n = 0;
    for ply in record.plies.iter().filter(|p| keep(p)) {
        let cp = ply.score_cp().unwrap();
        let white_cp = if ply.pos.turn() == Color::White { cp } else { -cp };
        let fen = Fen::from_position(&ply.pos, EnPassantMode::Legal);
        writeln!(out, "{fen} | {white_cp} | {result:.1}").unwrap();
        n += 1;
    }
    n
}

/// Whether a position teaches the net something true about its *static* value.
///
/// The net is asked to judge a position at a glance, so positions where the
/// search score depends on an immediate tactic are left out: the score there
/// reflects what happens after the capture or the check is resolved, not the
/// position as it stands. Mate scores are left out because they are not
/// centipawn values at all.
fn keep(ply: &PlyRecord) -> bool {
    let Some(cp) = ply.score_cp() else { return false };
    !ply.pos.is_check() && !ply.mv.is_capture() && !ply.mv.is_promotion() && cp.abs() < MATE_CP
}

fn parse_args() -> Args {
    let usage = || -> ! {
        eprintln!(
            "usage: datagen [--eval NAME] [--games N] [--nodes N] [--concurrency N] \
             [--random-every N] [--random-plies N] [--max-opening-score CP] [--seed N] [--hash MB] [--out FILE] [--json FILE]\nevaluators: {}",
            EVALUATORS.join(", ")
        );
        exit(2)
    };
    let mut args = Args {
        eval: "pst".to_string(),
        games: 1000,
        nodes: 5000,
        concurrency: std::thread::available_parallelism().map_or(1, |n| n.get()),
        opening: OpeningConfig::default(),
        // A new seed per run, so runs appending to one file play different openings.
        seed: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64),
        hash_mb: 16,
        out: "data.txt".to_string(),
        json: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--eval" => args.eval = value(),
            "--out" => args.out = value(),
            "--json" => args.json = Some(value()),
            "--games" => args.games = value().parse().unwrap_or_else(|_| usage()),
            "--nodes" => args.nodes = value().parse().unwrap_or_else(|_| usage()),
            "--concurrency" => args.concurrency = value().parse().unwrap_or_else(|_| usage()),
            "--random-every" => args.opening.random_every = value().parse().unwrap_or_else(|_| usage()),
            "--random-plies" => args.opening.random_plies = value().parse().unwrap_or_else(|_| usage()),
            "--max-opening-score" => args.opening.max_score = value().parse().unwrap_or_else(|_| usage()),
            "--seed" => args.seed = value().parse().unwrap_or_else(|_| usage()),
            "--hash" => args.hash_mb = value().parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    // Fails early, with a message, on an unknown name or an unreadable net.
    if let Err(e) = make_engine(&args.eval, 1) {
        eprintln!("{e}");
        exit(2)
    }
    args
}
