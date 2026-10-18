use shakmaty::{Chess, Color, Move, Position, Role};

use super::eval;

/// A static evaluation the search can be built on.
///
/// Every search thread owns its own copy (`Clone`), so an implementation may
/// keep mutable per-thread state. That state is what an NNUE needs: its first
/// layer (the "accumulator") is updated from the parent's on every move instead
/// of being recomputed from the whole board. The search tells the evaluator
/// about each move through `on_move` / `on_null_move`, indexed by `ply`, so an
/// implementation can keep one entry per ply and simply overwrite it — no undo
/// step is needed because every child at `ply + 1` is derived from `ply` again.
///
/// Stateless evaluators ignore the hooks.
pub trait Evaluator: Clone + Send + Sync + 'static {
    /// Rebuild all state from scratch for `pos`, which sits at `ply`.
    /// Called at the start of a search and when a helper thread joins a split
    /// point in the middle of the tree.
    fn reset(&mut self, pos: &Chess, ply: usize);

    /// The search played `mv` in `parent` (at `ply`); the child is at `ply + 1`.
    fn on_move(&mut self, ply: usize, parent: &Chess, mv: Move);

    /// The search passed (null move) at `ply`; the child is at `ply + 1`.
    fn on_null_move(&mut self, ply: usize) {
        let _ = ply;
    }

    /// Score of `pos` (at `ply`) in centipawns, from the side to move's view.
    fn evaluate(&mut self, pos: &Chess, ply: usize) -> i32;
}

/// The hand-written evaluation: material plus piece-square tables (`eval.rs`).
#[derive(Clone, Copy, Default)]
pub struct PstEval;

impl Evaluator for PstEval {
    fn reset(&mut self, _pos: &Chess, _ply: usize) {}

    fn on_move(&mut self, _ply: usize, _parent: &Chess, _mv: Move) {}

    #[inline]
    fn evaluate(&mut self, pos: &Chess, _ply: usize) -> i32 {
        eval::evaluate(pos)
    }
}

/// Material only, no piece-square tables. Deliberately weak: a baseline that
/// any real evaluator should beat clearly, and a sanity check for the tools
/// that compare evaluators.
#[derive(Clone, Copy, Default)]
pub struct MaterialEval;

impl Evaluator for MaterialEval {
    fn reset(&mut self, _pos: &Chess, _ply: usize) {}

    fn on_move(&mut self, _ply: usize, _parent: &Chess, _mv: Move) {}

    fn evaluate(&mut self, pos: &Chess, _ply: usize) -> i32 {
        let board = pos.board();
        let mut score = 0;
        for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            let white = (board.by_role(role) & board.by_color(Color::White)).count() as i32;
            let black = (board.by_role(role) & board.by_color(Color::Black)).count() as i32;
            score += eval::piece_value(role) * (white - black);
        }
        if pos.turn() == Color::White { score } else { -score }
    }
}
