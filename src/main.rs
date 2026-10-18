use std::process::exit;
use std::time::Duration;

use chess_engine::engine::{
    AnyEval,
    SearchEngine,
    ChessEngine,
    EVALUATORS,
    Limits,
    TimeMode
};
use chess_engine::play::{play_game};
use chess_engine::tui::TuiFrontend;

struct Args {
    eval: String,
    threads: usize,
    hash_mb: usize,
}

fn parse_args() -> Args {
    let usage = || -> ! {
        eprintln!(
            "usage: chess-engine [<evaluator>] [--threads N] [--hash MB]\n\
             evaluators: {}, or the path to a net (default pst)\n\
             --threads defaults to the core count, --hash to 1024 MB",
            EVALUATORS.join(", ")
        );
        exit(2)
    };
    let mut args = Args {
        eval: String::new(),
        threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        hash_mb: 1024,
    };
    let mut positional = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage());
        match arg.as_str() {
            "--threads" => args.threads = value(),
            "--hash" => args.hash_mb = value(),
            "-h" | "--help" => usage(),
            _ if arg.starts_with("--") => usage(),
            _ => positional.push(arg),
        }
    }
    match positional.len() {
        0 => {}
        1 => args.eval = positional.remove(0),
        _ => usage(),
    }
    if args.threads == 0 || args.hash_mb == 0 {
        usage();
    }
    args
}

fn main() -> std::io::Result<()> {
    // ~2 seconds of thinking per move.
    let args = parse_args();

    // Which evaluator to play against: `pst` (the default), `material`, or a
    // net, as `nnue:<path>` or just the path. Resolved before the terminal is
    // taken over, so an error is readable.
    let eval = if args.eval.is_empty() {
        AnyEval::default()
    } else {
        match AnyEval::parse_setting(&args.eval) {
            Ok(eval) => eval,
            Err(e) => {
                eprintln!("chess-engine: {e}");
                eprintln!("evaluators: {}, or the path to a net", EVALUATORS.join(", "));
                exit(2);
            }
        }
    };

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        hook(info);   // now the default handler prints to a sane terminal
    }));

    let mut engine = SearchEngine::with_evaluator(eval, args.hash_mb, args.threads);
    let handle = engine.search_handle();
    handle.set_limits(Limits::default().with_mode(TimeMode::Fixed(Duration::from_secs_f64(2.0))));
    let mut frontend = TuiFrontend::new()?;
    play_game(&mut frontend, &mut engine)?;
    // `frontend` restores the terminal on drop.
    Ok(())
}
