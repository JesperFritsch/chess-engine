
use chess_engine::{engine, render};
use shakmaty::{Chess};
use render::{TerminalRenderer, render_line};
use engine::{SearchContext, depth_bound_search};
// use shakmaty::{san::San};


fn main() {
    let pos = Chess::default();
    let renderer = TerminalRenderer;
    // let san: San = "e4".parse().unwrap();
    // let mv = san.to_move(&pos).unwrap();
    // let pos = pos.play(mv).unwrap();
    // renderer.render(&pos);
    let mut search_ctx = SearchContext::new(16, 1000);
    depth_bound_search(&mut search_ctx, &pos, 6).unwrap();
    render_line(&pos, &search_ctx.pv, &renderer);
}   