use chess_engine::play::{play_game, SearchEngine};
use chess_engine::tui::TuiFrontend;

fn main() -> std::io::Result<()> {
    // 64 MiB transposition table, ~2 seconds of thinking per move.
    let mut engine = SearchEngine::new(1024, 2000);
    let mut frontend = TuiFrontend::new()?;
    play_game(&mut frontend, &mut engine)?;
    // `frontend` restores the terminal on drop.
    Ok(())
}
