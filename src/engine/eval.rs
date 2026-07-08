use shakmaty::{Chess, Color, Position, Role, Square};

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

/// Static evaluation of `pos`, in centipawns, relative to the side to move
/// (positive = good for whoever is to move) — the sign convention negamax wants.
///
/// The score is material plus a piece-square-table (PST) bonus for every piece,
/// so the engine understands *where* a piece stands, not just that it exists. The
/// king's table is "tapered": interpolated between a midgame table (hide behind
/// pawns) and an endgame table (march to the centre) based on how much material
/// is left.
pub fn evaluate(pos: &Chess) -> i32 {
    let board = pos.board();

    // White's point of view; flipped for the side to move at the very end.
    let mut score = 0i32;
    let mut phase = 0i32;
    let mut white_king: Option<Square> = None;
    let mut black_king: Option<Square> = None;

    for (sq, piece) in board.clone().into_iter() {
        if piece.role == Role::King {
            // Kings carry no material value and their PST is phase-dependent, so
            // defer them until we know the game phase.
            match piece.color {
                Color::White => white_king = Some(sq),
                Color::Black => black_king = Some(sq),
            }
            continue;
        }

        let idx = pst_index(sq, piece.color);
        let value = piece_value(piece.role) + pst(piece.role, idx);
        score += if piece.color == Color::White { value } else { -value };
        phase += phase_weight(piece.role);
    }

    let phase = phase.min(TOTAL_PHASE);
    if let Some(sq) = white_king {
        score += king_pst(pst_index(sq, Color::White), phase);
    }
    if let Some(sq) = black_king {
        score -= king_pst(pst_index(sq, Color::Black), phase);
    }

    if pos.turn() == Color::White { score } else { -score }
}

/// Map a board square to an index into the piece-square tables below.
///
/// The tables are written from White's perspective with rank 8 on the first row
/// (so table index 0 = a8, index 63 = h1). A White piece therefore needs its
/// square's rank flipped (`^ 56`); a Black piece uses the square directly, which
/// mirrors it vertically — exactly the symmetry we want, since a black piece on
/// e7 should score like a white piece on e2.
fn pst_index(sq: Square, color: Color) -> usize {
    match color {
        Color::White => sq.to_usize() ^ 56,
        Color::Black => sq.to_usize(),
    }
}

fn pst(role: Role, idx: usize) -> i32 {
    match role {
        Role::Pawn => PAWN[idx],
        Role::Knight => KNIGHT[idx],
        Role::Bishop => BISHOP[idx],
        Role::Rook => ROOK[idx],
        Role::Queen => QUEEN[idx],
        Role::King => 0, // handled via king_pst
    }
}

/// Total phase units at the start of the game (see `phase_weight`).
const TOTAL_PHASE: i32 = 24;

/// Interpolate the king's PST between midgame and endgame. `phase` runs from
/// `TOTAL_PHASE` (all pieces on) down to 0 (bare endgame).
fn king_pst(idx: usize, phase: i32) -> i32 {
    (KING_MG[idx] * phase + KING_EG[idx] * (TOTAL_PHASE - phase)) / TOTAL_PHASE
}

/// How much each piece contributes to the "game phase". Summed over all pieces
/// on the board it is 24 at the start and drops toward 0 as pieces come off.
fn phase_weight(role: Role) -> i32 {
    match role {
        Role::Knight | Role::Bishop => 1,
        Role::Rook => 2,
        Role::Queen => 4,
        _ => 0,
    }
}

// --- Piece-square tables (Michniewski "Simplified Evaluation"), a8 = index 0 ---

#[rustfmt::skip]
const PAWN: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    fn pos(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap()
    }

    #[test]
    fn start_position_is_symmetric() {
        // A perfectly symmetric position must evaluate to 0.
        assert_eq!(evaluate(&Chess::default()), 0);
    }

    #[test]
    fn central_knight_beats_rim_knight() {
        // White knight on e5 (central) vs on a3 (rim), otherwise identical.
        let central = pos("4k3/8/8/4N3/8/8/8/4K3 w - - 0 1");
        let rim = pos("4k3/8/8/8/8/N7/8/4K3 w - - 0 1");
        assert!(
            evaluate(&central) > evaluate(&rim),
            "central {} should beat rim {}",
            evaluate(&central),
            evaluate(&rim)
        );
    }

    #[test]
    fn king_centralizes_in_the_endgame() {
        // With almost nothing on the board, a central king should score better
        // than a cornered one (endgame king table dominates). Black king kept
        // far away so the positions are legal (kings never adjacent).
        let central = pos("8/8/4k3/8/4K3/8/8/8 w - - 0 1");
        let corner = pos("8/8/4k3/8/8/8/8/K7 w - - 0 1");
        assert!(evaluate(&central) > evaluate(&corner));
    }
}
