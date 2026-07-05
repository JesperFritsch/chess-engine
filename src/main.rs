
use chess_engine::{engine, render};
use shakmaty::{Chess, Position};
use render::{TerminalRenderer, Renderer, render_line};

// use shakmaty::{san::San};


fn main() {
    let pos = Chess::default();
    let renderer = TerminalRenderer;
    // let san: San = "e4".parse().unwrap();
    // let mv = san.to_move(&pos).unwrap();
    // let pos = pos.play(mv).unwrap();
    // renderer.render(&pos);
    let search_result = engine::best_line(&pos, 6);
    let mv = search_result.line.first().unwrap().clone();
    let pos = pos.play(mv).unwrap();
    renderer.render(&pos);
    render_line(&pos, &search_result.line[1..], &renderer);
}   