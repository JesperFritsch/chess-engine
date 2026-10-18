//! Static evaluation of positions, without a search.
//!
//!   echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | evalfen nnue:nets/net.bin
//!
//! Reads one position per line from stdin (a FEN, optionally followed by
//! `| ...` as in datagen files) and prints its evaluation in centipawns from
//! the side to move's point of view, one per line. `nnue/loop.py` uses it to
//! check that the engine computes the same values as the trainer.

use chess_engine::engine::{nnue, Evaluator, MaterialEval, PstEval};
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess};
use std::io::{BufRead, Write};
use std::process::exit;

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: evalfen <pst | material | nnue:<net file>>  (FENs on stdin)");
        exit(2)
    });
    match name.as_str() {
        "pst" => run(PstEval),
        "material" => run(MaterialEval),
        _ => match name.strip_prefix("nnue:") {
            Some(path) => run(nnue::load_evaluator(path).unwrap_or_else(|e| {
                eprintln!("{e}");
                exit(1)
            })),
            None => {
                eprintln!("unknown evaluator {name:?}");
                exit(2)
            }
        },
    }
}

fn run<E: Evaluator>(mut eval: E) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap_or_else(|e| {
            eprintln!("read error: {e}");
            exit(1)
        });
        let fen = line.split('|').next().unwrap().trim();
        if fen.is_empty() {
            continue;
        }
        let pos: Chess = match fen.parse::<Fen>().ok().and_then(|f| f.into_position(CastlingMode::Standard).ok()) {
            Some(pos) => pos,
            None => {
                eprintln!("cannot parse FEN: {fen}");
                exit(1)
            }
        };
        eval.reset(&pos, 0);
        writeln!(out, "{}", eval.evaluate(&pos, 0)).unwrap();
    }
}
