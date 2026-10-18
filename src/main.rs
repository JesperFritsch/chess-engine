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

fn main() -> std::io::Result<()> {
    // 64 MiB transposition table, ~2 seconds of thinking per move.

    // Which evaluator to play against: `pst` (the default), `material`, or a
    // net, as `nnue:<path>` or just the path. Checked before the terminal is
    // taken over, so an error is readable.
    let eval = match std::env::args().nth(1) {
        None => AnyEval::default(),
        Some(name) => match AnyEval::parse_setting(&name) {
            Ok(eval) => eval,
            Err(e) => {
                eprintln!("chess-engine: {e}");
                eprintln!("usage: chess-engine [{}|<net file>]", EVALUATORS.join("|"));
                std::process::exit(2);
            }
        },
    };

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        hook(info);   // now the default handler prints to a sane terminal
    }));

    let mut engine = SearchEngine::with_evaluator(eval, 1024, 1);
    let handle = engine.search_handle();
    handle.set_limits(Limits::default().with_mode(TimeMode::Fixed(Duration::from_secs_f64(2.0)))); 
    let mut frontend = TuiFrontend::new()?;
    play_game(&mut frontend, &mut engine)?;
    // `frontend` restores the terminal on drop.
    Ok(())
}
