
use shakmaty::{Chess, Move, Position};

pub mod terminal;

pub use terminal::TerminalRenderer;

pub trait Renderer {
    fn render(&self, pos: &Chess);
}

pub fn render_line(pos: &Chess, moves: &[Move], renderer: &impl Renderer) {
    let mut current_pos = pos.clone();
    println!("line of {} moves:", moves.len());
    renderer.render(&current_pos);
    for m in moves {
        current_pos = current_pos.play(*m).unwrap();
        println!();
        renderer.render(&current_pos);
    }
}