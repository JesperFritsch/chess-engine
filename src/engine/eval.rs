use shakmaty::{Chess, Position, Role};

pub const fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 0,
    }
}

pub fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();
    let mut score = 0;
    for (_sq, piece) in board.clone().into_iter() {
        let v = piece_value(piece.role);
        score += if piece.color == pos.turn() { v } else { -v };
    }
    score
}
