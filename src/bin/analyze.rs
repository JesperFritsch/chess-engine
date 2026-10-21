//! Searches a single position and shows the search as it progresses.
//!
//!   echo "position startpos moves e2e4" | uci2fen | analyze --eval nnue:nets/net.bin --depth 18
//!   analyze --depth 18 "8/8/1r4p1/4Np1p/4kP1P/6P1/8/6K1 b - - 91 185"
//!
//! Takes the FEN as an argument, or reads it from stdin. With no `--depth`, `--nodes` or `--time` the search
//! is unbound: it runs until it finds a mate or is interrupted. The info for
//! the latest iteration replaces the previous one when stdout is a terminal,
//! and is printed line by line otherwise.

use chess_engine::engine::{AnyEval, ChessEngine, Limits, SearchEngine, SearchProgress, TimeMode, EVALUATORS};
use crossterm::terminal;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Move};
use std::io::{IsTerminal, Write};
use std::process::exit;
use std::time::Duration;

struct Args {
    eval: String,
    depth: Option<u8>,
    nodes: Option<u64>,
    time_ms: Option<u64>,
    threads: usize,
    hash_mb: usize,
    fen: Option<String>,
}

fn main() {
    let args = parse_args();
    let eval = AnyEval::parse(&args.eval).unwrap_or_else(|e| {
        eprintln!("{e}");
        exit(2)
    });

    let line = args.fen.clone().unwrap_or_else(|| {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap_or_else(|e| {
            eprintln!("read error: {e}");
            exit(1)
        });
        line
    });
    let fen = line.split('|').next().unwrap().trim();
    let pos: Chess = fen
        .parse::<Fen>()
        .ok()
        .and_then(|f| f.into_position(CastlingMode::Standard).ok())
        .unwrap_or_else(|| {
            eprintln!("cannot parse FEN: {fen}");
            exit(1)
        });

    let mut engine = SearchEngine::with_evaluator(eval, args.hash_mb, args.threads);
    engine.search_handle().set_limits(Limits {
        time_mode: match args.time_ms {
            Some(ms) => TimeMode::Fixed(Duration::from_millis(ms)),
            None => TimeMode::Unbound,
        },
        max_depth: args.depth,
        max_nodes: args.nodes,
        restrict_to: None,
    });
    engine.set_position(pos, &[]);

    let mut display = Display::new();
    println!("fen  {fen}");
    println!("eval {}  threads {}  hash {} MiB", args.eval, args.threads, args.hash_mb);
    let result = engine.search(&mut |p| display.show(p));

    let best = result.best_move.as_ref().map_or("(none)".to_string(), uci);
    let score = result.score.map_or("-".to_string(), |s| s.to_string());
    println!("bestmove {best}  score {score}  depth {}  nodes {}", result.depth, result.nodes);
}

/// Redraws the info block in place, by moving the cursor back over the lines
/// the previous block took up (counting lines the terminal wrapped).
struct Display {
    overwrite: bool,
    lines: u16,
}

impl Display {
    fn new() -> Self {
        Display { overwrite: std::io::stdout().is_terminal(), lines: 0 }
    }

    fn show(&mut self, p: &SearchProgress) {
        let pv: Vec<String> = p.pv.iter().map(uci).collect();
        let text = [
            format!("depth    {}/{}", p.depth, p.seldepth),
            format!("score    {}", p.score),
            format!("nodes    {}  ({} n/s)", p.nodes, p.nodes_per_s),
            format!("time     {:.2}s", p.elapsed.as_secs_f64()),
            format!("hashfull {}‰", p.hashfull),
            format!("pv       {}", pv.join(" ")),
        ];
        let mut out = std::io::stdout().lock();
        if !self.overwrite {
            writeln!(out, "{}", text.join("  |  ")).unwrap();
            return;
        }
        if self.lines > 0 {
            // Up to the start of the previous block, then clear to the end of the screen.
            write!(out, "\x1b[{}F\x1b[J", self.lines).unwrap();
        }
        let width = terminal::size().map_or(80, |(w, _)| w.max(1)) as usize;
        self.lines = 0;
        for line in &text {
            writeln!(out, "{line}").unwrap();
            self.lines += line.chars().count().max(1).div_ceil(width) as u16;
        }
        out.flush().unwrap();
    }
}

fn uci(m: &Move) -> String {
    m.to_uci(CastlingMode::Standard).to_string()
}

fn parse_args() -> Args {
    let usage = || -> ! {
        eprintln!(
            "usage: analyze [--eval NAME] [--depth N] [--nodes N] [--time MS] [--threads N] [--hash MB] [FEN]  (FEN on stdin if not given)\n\
             evaluators: {}\nwith no --depth, --nodes or --time the search is unbound",
            EVALUATORS.join(", ")
        );
        exit(2)
    };
    let mut args = Args {
        eval: "pst".to_string(),
        depth: None,
        nodes: None,
        time_ms: None,
        threads: 1,
        hash_mb: 16,
        fen: None,
    };
    // A FEN is six fields; unquoted, they arrive as separate arguments.
    let mut fen_parts = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        if arg == "--eval" {
            args.eval = it.next().unwrap_or_else(|| usage());
            continue;
        }
        let mut value = || it.next().and_then(|v| v.parse::<u64>().ok()).unwrap_or_else(|| usage());
        match arg.as_str() {
            "--depth" => args.depth = Some(value().try_into().unwrap_or_else(|_| usage())),
            "--nodes" => args.nodes = Some(value()),
            "--time" => args.time_ms = Some(value()),
            "--threads" => args.threads = value() as usize,
            "--hash" => args.hash_mb = value() as usize,
            _ if arg.starts_with("--") => usage(),
            _ => fen_parts.push(arg),
        }
    }
    if !fen_parts.is_empty() {
        args.fen = Some(fen_parts.join(" "));
    }
    args
}
