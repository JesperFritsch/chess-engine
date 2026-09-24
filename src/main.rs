use std::time::Duration;

use chess_engine::engine::{
    SearchEngine, 
    ChessEngine,
    Limits,
    TimeMode
};
use chess_engine::play::{play_game};
use chess_engine::tui::TuiFrontend;

fn main() -> std::io::Result<()> {
    // 64 MiB transposition table, ~2 seconds of thinking per move.

    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
        hook(info);   // now the default handler prints to a sane terminal
    }));

    let mut engine = SearchEngine::new(1024);
    let handle = engine.search_handle();
    handle.set_limits(Limits::default().with_mode(TimeMode::Fixed(Duration::from_secs_f64(2.0)))); 
    let mut frontend = TuiFrontend::new()?;
    play_game(&mut frontend, &mut engine)?;
    // `frontend` restores the terminal on drop.
    Ok(())
}
