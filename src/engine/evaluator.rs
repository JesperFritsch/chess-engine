use shakmaty::{Chess, Color, Move, Position, Role};

use super::eval;
use super::nnue::{self, NnueEval};

/// The evaluator names accepted on a command line and by the `EvalFile` UCI
/// option, for usage and error messages.
pub const EVALUATORS: &[&str] = &["pst", "material", "nnue:<net file>"];

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

/// Any of the engine's evaluators, picked at run time.
///
/// [`SearchEngine`](crate::engine::SearchEngine) is generic over its evaluator
/// so that the eval call can be inlined, which fixes the choice when the
/// binary is compiled. The UCI engine and the TUI cannot work that way: which
/// evaluator to use arrives as an option or an argument, long after
/// compilation. This enum is the one place that pays for the flexibility, and
/// it pays in a branch per call rather than in the indirect call a
/// `dyn Evaluator` would cost. `arena` and `datagen` keep using the concrete
/// types, so nothing about self-play or matches changes.
#[derive(Clone)]
pub enum AnyEval {
    Pst(PstEval),
    Material(MaterialEval),
    Nnue(NnueEval),
}

impl Default for AnyEval {
    fn default() -> Self {
        AnyEval::Pst(PstEval)
    }
}

impl AnyEval {
    /// Exactly one of `pst`, `material` or `nnue:<path>`. This is the spelling
    /// the tools take, where a typo should be an error rather than a guess.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "pst" => Ok(AnyEval::Pst(PstEval)),
            "material" => Ok(AnyEval::Material(MaterialEval)),
            _ => match name.strip_prefix("nnue:") {
                Some(path) => Ok(AnyEval::Nnue(nnue::load_evaluator(path)?)),
                None => Err(format!(
                    "unknown evaluator {name:?}; known: {}",
                    EVALUATORS.join(", ")
                )),
            },
        }
    }

    /// As [`AnyEval::parse`], but for a value a person or a GUI supplied, where
    /// being forgiving is worth more than catching typos:
    ///
    /// * empty (some GUIs send the literal `<empty>`) or `none` means the
    ///   built-in evaluation, so clearing the option goes back to PST;
    /// * anything that is not a known name is taken to be the path to a net,
    ///   so `EvalFile` can be set to a plain file name as it is in other
    ///   engines.
    pub fn parse_setting(value: &str) -> Result<Self, String> {
        match value.trim() {
            "" | "<empty>" | "none" => Ok(AnyEval::default()),
            name @ ("pst" | "material") => Self::parse(name),
            name if name.starts_with("nnue:") => Self::parse(name),
            path => Ok(AnyEval::Nnue(nnue::load_evaluator(path)?)),
        }
    }

    /// Which evaluator this is, for `info string` and the usage text.
    pub fn name(&self) -> &'static str {
        match self {
            AnyEval::Pst(_) => "pst",
            AnyEval::Material(_) => "material",
            AnyEval::Nnue(_) => "nnue",
        }
    }
}

impl Evaluator for AnyEval {
    fn reset(&mut self, pos: &Chess, ply: usize) {
        match self {
            AnyEval::Pst(e) => e.reset(pos, ply),
            AnyEval::Material(e) => e.reset(pos, ply),
            AnyEval::Nnue(e) => e.reset(pos, ply),
        }
    }

    #[inline]
    fn on_move(&mut self, ply: usize, parent: &Chess, mv: Move) {
        match self {
            AnyEval::Pst(e) => e.on_move(ply, parent, mv),
            AnyEval::Material(e) => e.on_move(ply, parent, mv),
            AnyEval::Nnue(e) => e.on_move(ply, parent, mv),
        }
    }

    #[inline]
    fn on_null_move(&mut self, ply: usize) {
        match self {
            AnyEval::Pst(e) => e.on_null_move(ply),
            AnyEval::Material(e) => e.on_null_move(ply),
            AnyEval::Nnue(e) => e.on_null_move(ply),
        }
    }

    #[inline]
    fn evaluate(&mut self, pos: &Chess, ply: usize) -> i32 {
        match self {
            AnyEval::Pst(e) => e.evaluate(pos, ply),
            AnyEval::Material(e) => e.evaluate(pos, ply),
            AnyEval::Nnue(e) => e.evaluate(pos, ply),
        }
    }
}
