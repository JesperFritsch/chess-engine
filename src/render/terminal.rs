
use owo_colors::OwoColorize;

use super::Renderer;
pub struct TerminalRenderer;


use shakmaty::{Color, Role, Chess, Position, Rank, File as ChessFile};


fn render_piece(color: Color, role: Role) -> String {
    let piece_str = role.upper_char().to_string();

    match color {
        Color::White => piece_str.white().to_string(),
        Color::Black => piece_str.red().to_string(),
    }
}

impl Renderer for TerminalRenderer {
    fn render(&self, pos: &Chess) {
        for rank in Rank::ALL.into_iter().rev() {
            for file in ChessFile::ALL {
                let square = shakmaty::Square::from_coords(file, rank);
                if let Some(piece) = pos.board().piece_at(square) {
                    let piece_str = render_piece(piece.color, piece.role);
                    print!("{} ", piece_str);
                } else {
                    print!(". ");
                }
            }
            println!();
        }  
    }
}