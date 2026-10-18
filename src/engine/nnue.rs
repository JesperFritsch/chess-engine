//! The NNUE evaluator: the network trained by `nnue/train.py`.
//!
//! The network, for each position:
//!
//!   white view: 768 inputs -> 256 accumulator \
//!                                               > [side to move | other side] -> CReLU -> 1 output
//!   black view: 768 inputs -> 256 accumulator /
//!
//! An input is one piece on one square, so a position switches on about 30 of
//! the 768. The first layer's output, the accumulator, is therefore just the
//! bias plus one weight row per piece. A move changes only two to four pieces,
//! so instead of re-adding ~30 rows per evaluation, the search's move hooks
//! update the parent's accumulator by those few rows. That is the "efficiently
//! updatable" in NNUE, and why the evaluator keeps one accumulator per ply.
//!
//! Everything is integer arithmetic on the quantized weights `train.py` exports;
//! the constants below must match it.

use shakmaty::{Chess, Color, File, Move, Position, Role, Square};
use std::sync::Arc;

use super::evaluator::Evaluator;

const FEATURES: usize = 768;
pub const HIDDEN: usize = 256;
/// Accumulator values are in units of 1/QA, output weights in units of 1/QB.
const QA: i32 = 255;
const QB: i32 = 64;
/// Centipawns per unit of network output.
const SCALE: i32 = 400;
/// One accumulator per ply a search can reach. `ply` is a `u8`, and a child
/// sits at `ply + 1`.
const STACK: usize = u8::MAX as usize + 2;

/// The trained weights. Read-only, so one copy is shared by every thread.
pub struct Network {
    /// `ft_weights[feature]` is the weight row that feature adds to the accumulator.
    ft_weights: Vec<[i16; HIDDEN]>,
    ft_bias: [i16; HIDDEN],
    /// The side to move's half first, then the other side's.
    out_weights: [i16; 2 * HIDDEN],
    out_bias: i32,
}

impl Network {
    pub fn load(path: &str) -> Result<Network, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read net {path}: {e}"))?;
        Network::from_bytes(&bytes).map_err(|e| format!("bad net {path}: {e}"))
    }

    /// Parses the format written by `export` in `nnue/train.py`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Network, String> {
        let expected = 12 + 2 * (FEATURES * HIDDEN + HIDDEN + 2 * HIDDEN) + 4;
        if bytes.len() != expected {
            return Err(format!("{} bytes, expected {expected} for hidden size {HIDDEN}", bytes.len()));
        }
        if &bytes[0..4] != b"NNUE" {
            return Err("missing NNUE header".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        if u32_at(4) != 1 {
            return Err(format!("unsupported version {}", u32_at(4)));
        }
        if u32_at(8) as usize != HIDDEN {
            return Err(format!("hidden size {}, engine is built for {HIDDEN}", u32_at(8)));
        }

        let mut i16s = bytes[12..expected - 4].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]));
        let mut fill = |out: &mut [i16]| out.iter_mut().for_each(|v| *v = i16s.next().unwrap());
        let mut ft_weights = vec![[0i16; HIDDEN]; FEATURES];
        ft_weights.iter_mut().for_each(|row| fill(row));
        let mut ft_bias = [0i16; HIDDEN];
        fill(&mut ft_bias);
        let mut out_weights = [0i16; 2 * HIDDEN];
        fill(&mut out_weights);
        let out_bias = i32::from_le_bytes(bytes[expected - 4..].try_into().unwrap());

        Ok(Network { ft_weights, ft_bias, out_weights, out_bias })
    }
}

/// The first layer's output for both views of one position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(64))]
struct Accumulator {
    /// Indexed by view: 0 = White's, 1 = Black's.
    views: [[i16; HIDDEN]; 2],
}

/// The input index of `color`'s `role` on `sq`, as seen from `view`'s side.
///
/// White's view is the board as it stands; Black's view is the board flipped
/// vertically with the colors swapped, so a feature always means "my piece" or
/// "their piece" rather than "White's" or "Black's". Matches `pack` and `flip`
/// in `train.py`.
#[inline]
fn feature(view: Color, color: Color, role: Role, sq: Square) -> usize {
    let role = role as usize - 1;
    match view {
        Color::White => (color == Color::Black) as usize * 384 + role * 64 + sq as usize,
        Color::Black => (color == Color::White) as usize * 384 + role * 64 + (sq as usize ^ 56),
    }
}

#[derive(Clone)]
pub struct NnueEval {
    net: Arc<Network>,
    /// `stack[ply]` is the accumulator of the position at `ply`. Empty until the
    /// first `reset`, so cloning an evaluator that has not searched is cheap.
    stack: Vec<Accumulator>,
}

impl NnueEval {
    pub fn new(net: Arc<Network>) -> NnueEval {
        NnueEval { net, stack: Vec::new() }
    }

    /// The accumulator of `pos`, built from scratch.
    fn refresh(&self, pos: &Chess) -> Accumulator {
        let mut acc = Accumulator { views: [self.net.ft_bias; 2] };
        for (sq, piece) in pos.board().clone() {
            for (v, view) in [Color::White, Color::Black].into_iter().enumerate() {
                add(&mut acc.views[v], &self.net.ft_weights[feature(view, piece.color, piece.role, sq)]);
            }
        }
        acc
    }

    fn output(&self, acc: &Accumulator, turn: Color) -> i32 {
        let (us, them) = match turn {
            Color::White => (&acc.views[0], &acc.views[1]),
            Color::Black => (&acc.views[1], &acc.views[0]),
        };
        let (w_us, w_them) = self.net.out_weights.split_at(HIDDEN);
        // CReLU: clamp to [0, QA], i.e. [0, 1] in real units.
        let dot = |acc: &[i16; HIDDEN], w: &[i16]| -> i32 {
            acc.iter().zip(w).map(|(&a, &w)| (a as i32).clamp(0, QA) * w as i32).sum()
        };
        let sum = dot(us, w_us) + dot(them, w_them) + self.net.out_bias;
        sum * SCALE / (QA * QB)
    }

    /// Static evaluation of a standalone position, without a search. For tools
    /// and tests.
    pub fn evaluate_position(&self, pos: &Chess) -> i32 {
        self.output(&self.refresh(pos), pos.turn())
    }
}

#[inline]
fn add(acc: &mut [i16; HIDDEN], row: &[i16; HIDDEN]) {
    acc.iter_mut().zip(row).for_each(|(a, &w)| *a += w);
}

#[inline]
fn sub(acc: &mut [i16; HIDDEN], row: &[i16; HIDDEN]) {
    acc.iter_mut().zip(row).for_each(|(a, &w)| *a -= w);
}

impl Evaluator for NnueEval {
    fn reset(&mut self, pos: &Chess, ply: usize) {
        if self.stack.is_empty() {
            self.stack = vec![Accumulator { views: [[0; HIDDEN]; 2] }; STACK];
        }
        self.stack[ply] = self.refresh(pos);
    }

    fn on_move(&mut self, ply: usize, parent: &Chess, mv: Move) {
        let us = parent.turn();
        // The pieces that leave a square and the ones that arrive on one.
        let mut removed: [(Color, Role, Square); 2] = [(us, Role::Pawn, Square::A1); 2];
        let mut added: [(Color, Role, Square); 2] = [(us, Role::Pawn, Square::A1); 2];
        let (n_removed, n_added) = match mv {
            Move::Normal { role, from, capture, to, promotion } => {
                removed[0] = (us, role, from);
                added[0] = (us, promotion.unwrap_or(role), to);
                match capture {
                    Some(captured) => {
                        removed[1] = (!us, captured, to);
                        (2, 1)
                    }
                    None => (1, 1),
                }
            }
            Move::EnPassant { from, to } => {
                removed[0] = (us, Role::Pawn, from);
                // The captured pawn stands beside `from`, on the file of `to`.
                removed[1] = (!us, Role::Pawn, Square::from_coords(to.file(), from.rank()));
                added[0] = (us, Role::Pawn, to);
                (2, 1)
            }
            Move::Castle { king, rook } => {
                // shakmaty encodes castling as "king takes own rook".
                let (king_to, rook_to) = if rook.file() > king.file() { (File::G, File::F) } else { (File::C, File::D) };
                removed = [(us, Role::King, king), (us, Role::Rook, rook)];
                added = [
                    (us, Role::King, Square::from_coords(king_to, king.rank())),
                    (us, Role::Rook, Square::from_coords(rook_to, king.rank())),
                ];
                (2, 2)
            }
            Move::Put { .. } => unreachable!("drops do not exist in standard chess"),
        };

        let (parents, children) = self.stack.split_at_mut(ply + 1);
        let child = &mut children[0];
        *child = parents[ply];
        for (v, view) in [Color::White, Color::Black].into_iter().enumerate() {
            for &(color, role, sq) in &removed[..n_removed] {
                sub(&mut child.views[v], &self.net.ft_weights[feature(view, color, role, sq)]);
            }
            for &(color, role, sq) in &added[..n_added] {
                add(&mut child.views[v], &self.net.ft_weights[feature(view, color, role, sq)]);
            }
        }
    }

    fn on_null_move(&mut self, ply: usize) {
        // Nothing moves; only the side to move changes, which `evaluate` handles.
        self.stack[ply + 1] = self.stack[ply];
    }

    fn evaluate(&mut self, pos: &Chess, ply: usize) -> i32 {
        let acc = &self.stack[ply];
        // Any slip in the incremental updates shows up here, in debug builds.
        debug_assert_eq!(*acc, self.refresh(pos), "NNUE accumulator out of sync at ply {ply}");
        self.output(acc, pos.turn())
    }
}

/// Convenience for building an engine: load a net and wrap it.
pub fn load_evaluator(path: &str) -> Result<NnueEval, String> {
    Ok(NnueEval::new(Arc::new(Network::load(path)?)))
}
