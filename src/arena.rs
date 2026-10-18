//! Engine-vs-engine games: the shared machinery behind evaluator matches and
//! self-play data generation.
//!
//! Nothing here knows *why* a game is played. [`play_game`] plays one game and
//! records every move with the score the engine gave it; [`run_games`] plays
//! many in parallel and hands each finished [`GameRecord`] to a callback. A
//! match tallies the results; data generation writes the positions to disk.

use crate::engine::{ChessEngine, Limits, Score, SearchResult};
use shakmaty::zobrist::Zobrist64;
use shakmaty::{Chess, Color, EnPassantMode, Move, Position};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

/// A mate score in centipawns, for adjudication and for training labels.
/// Anything this large or larger means "won", whatever the distance to mate.
pub const MATE_CP: i32 = 30_000;

/// How a game is set up and when it is cut short.
#[derive(Clone)]
pub struct GameConfig {
    /// Search limits for every move, typically `max_nodes` with `TimeMode::Unbound`.
    pub limits: Limits,
    pub adjudication: Adjudication,
}

/// Ending games early once the outcome is clear. Both rules look at the scores
/// of consecutive plies, so the two engines have to agree, not just one.
#[derive(Clone)]
pub struct Adjudication {
    /// Win once |score| >= `win_score` (same side ahead) for `win_plies` plies.
    pub win_score: i32,
    pub win_plies: usize,
    /// Draw once |score| <= `draw_score` for `draw_plies` plies, from `draw_after_ply` on.
    pub draw_score: i32,
    pub draw_plies: usize,
    pub draw_after_ply: usize,
    /// Draw a game that reaches this many plies (counting the opening).
    pub max_plies: usize,
}

impl Default for Adjudication {
    fn default() -> Self {
        Adjudication {
            win_score: 1000,
            win_plies: 4,
            draw_score: 10,
            draw_plies: 8,
            draw_after_ply: 80,
            max_plies: 400,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw,
}

impl GameResult {
    /// The result for `color`: 1.0 win, 0.5 draw, 0.0 loss.
    pub fn score_for(self, color: Color) -> f64 {
        match (self, color) {
            (GameResult::Draw, _) => 0.5,
            (GameResult::WhiteWins, Color::White) | (GameResult::BlackWins, Color::Black) => 1.0,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Termination {
    Checkmate,
    Stalemate,
    InsufficientMaterial,
    FiftyMoves,
    Repetition,
    AdjudicatedWin,
    AdjudicatedDraw,
    MaxPlies,
}

/// The starting point of a game: a position plus moves already played from it.
/// The moves are part of the game's history (repetitions count them) but are
/// not recorded as searched plies.
#[derive(Clone)]
pub struct Opening {
    pub start: Chess,
    pub moves: Vec<Move>,
}

impl Opening {
    pub fn position(&self) -> Chess {
        let mut pos = self.start.clone();
        for &m in &self.moves {
            pos.play_unchecked(m);
        }
        pos
    }
}

/// One searched move.
#[derive(Clone)]
pub struct PlyRecord {
    /// The position before the move.
    pub pos: Chess,
    pub mv: Move,
    /// From the side to move's point of view, as the engine reported it.
    pub score: Option<Score>,
    pub depth: u8,
    pub nodes: u64,
}

impl PlyRecord {
    /// The score in centipawns from the side to move's view; mates become ±[`MATE_CP`].
    pub fn score_cp(&self) -> Option<i32> {
        self.score.as_ref().map(score_to_cp)
    }
}

#[derive(Clone)]
pub struct GameRecord {
    pub opening: Opening,
    /// Every move after the opening, in order.
    pub plies: Vec<PlyRecord>,
    pub result: GameResult,
    pub termination: Termination,
}

pub fn score_to_cp(score: &Score) -> i32 {
    match *score {
        Score::Cp(cp) => cp,
        Score::Mate(n) if n > 0 => MATE_CP,
        Score::Mate(_) => -MATE_CP,
    }
}

/// Play one game between two engines from `opening`. Both engines are cleared
/// first, so nothing carries over from an earlier game.
pub fn play_game(
    white: &mut dyn ChessEngine,
    black: &mut dyn ChessEngine,
    opening: &Opening,
    config: &GameConfig,
) -> GameRecord {
    white.clear();
    black.clear();
    white.search_handle().set_limits(config.limits.clone());
    black.search_handle().set_limits(config.limits.clone());

    let adj = &config.adjudication;
    let mut pos = opening.start.clone();
    let mut keys = vec![key(&pos)];
    for &m in &opening.moves {
        pos.play_unchecked(m);
        keys.push(key(&pos));
    }
    let mut history = opening.moves.clone();
    let mut plies = Vec::new();
    // Consecutive plies the adjudication conditions have held, and for the win
    // rule, which side was ahead.
    let mut win_run = 0usize;
    let mut win_side = Color::White;
    let mut draw_run = 0usize;

    let (result, termination) = loop {
        if let Some(end) = natural_end(&pos, &keys) {
            break end;
        }
        if history.len() >= adj.max_plies {
            break (GameResult::Draw, Termination::MaxPlies);
        }

        let engine: &mut dyn ChessEngine = if pos.turn() == Color::White { &mut *white } else { &mut *black };
        engine.set_position(opening.start.clone(), &history);
        let SearchResult { best_move, score, depth, nodes, .. } = engine.search(&mut |_| {});
        // The position has legal moves (checked above), so a search always has one.
        let mv = best_move.expect("engine returned no move in a position with legal moves");

        if let Some(cp) = score.as_ref().map(score_to_cp) {
            // Adjudication works in White's terms, so both engines' scores line up.
            let white_cp = if pos.turn() == Color::White { cp } else { -cp };
            let ahead = if white_cp > 0 { Color::White } else { Color::Black };
            if white_cp.abs() >= adj.win_score {
                win_run = if win_run > 0 && ahead == win_side { win_run + 1 } else { 1 };
                win_side = ahead;
            } else {
                win_run = 0;
            }
            draw_run = if white_cp.abs() <= adj.draw_score { draw_run + 1 } else { 0 };
        }

        plies.push(PlyRecord { pos: pos.clone(), mv, score, depth, nodes });
        pos.play_unchecked(mv);
        keys.push(key(&pos));
        history.push(mv);

        if win_run >= adj.win_plies {
            let result = if win_side == Color::White { GameResult::WhiteWins } else { GameResult::BlackWins };
            break (result, Termination::AdjudicatedWin);
        }
        if draw_run >= adj.draw_plies && history.len() >= adj.draw_after_ply {
            break (GameResult::Draw, Termination::AdjudicatedDraw);
        }
    };

    GameRecord { opening: opening.clone(), plies, result, termination }
}

/// The game is over by the rules of chess.
fn natural_end(pos: &Chess, keys: &[Zobrist64]) -> Option<(GameResult, Termination)> {
    if pos.legal_moves().is_empty() {
        return Some(if pos.is_check() {
            let result = if pos.turn() == Color::White { GameResult::BlackWins } else { GameResult::WhiteWins };
            (result, Termination::Checkmate)
        } else {
            (GameResult::Draw, Termination::Stalemate)
        });
    }
    if pos.is_insufficient_material() {
        return Some((GameResult::Draw, Termination::InsufficientMaterial));
    }
    if pos.halfmoves() >= 100 {
        return Some((GameResult::Draw, Termination::FiftyMoves));
    }
    let current = *keys.last().unwrap();
    if keys.iter().filter(|&&k| k == current).count() >= 3 {
        return Some((GameResult::Draw, Termination::Repetition));
    }
    None
}

fn key(pos: &Chess) -> Zobrist64 {
    pos.zobrist_hash(EnPassantMode::Legal)
}

/// `plies` uniformly random legal moves from the starting position. Retries
/// until the game is still going afterwards.
pub fn random_opening(rng: &mut Rng, plies: usize) -> Opening {
    'retry: loop {
        let mut pos = Chess::default();
        let mut moves = Vec::with_capacity(plies);
        for _ in 0..plies {
            let legal = pos.legal_moves();
            if legal.is_empty() {
                continue 'retry;
            }
            let m = legal[rng.below(legal.len())];
            pos.play_unchecked(m);
            moves.push(m);
        }
        if pos.legal_moves().is_empty() {
            continue;
        }
        return Opening { start: Chess::default(), moves };
    }
}

/// One game for [`run_games`]: the opening, and which of the two engines a
/// thread holds plays White.
#[derive(Clone)]
pub struct Job {
    pub opening: Opening,
    /// `true`: the first engine from `make_engines` plays White.
    pub first_is_white: bool,
}

/// Play `jobs` on `concurrency` threads.
///
/// Every thread calls `make_engines` once and reuses the pair for all its games,
/// so engines are built per thread and never shared. `on_game` runs on the
/// calling thread, in completion order, with the job's index; it can therefore
/// write files or print without any locking.
pub fn run_games<M>(
    jobs: &[Job],
    concurrency: usize,
    config: &GameConfig,
    make_engines: M,
    mut on_game: impl FnMut(usize, &Job, GameRecord),
) where
    M: Fn() -> (Box<dyn ChessEngine>, Box<dyn ChessEngine>) + Sync,
{
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel();
    std::thread::scope(|scope| {
        for _ in 0..concurrency.max(1) {
            let tx = tx.clone();
            let (next, make_engines) = (&next, &make_engines);
            scope.spawn(move || {
                let (mut first, mut second) = make_engines();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let record = if job.first_is_white {
                        play_game(first.as_mut(), second.as_mut(), &job.opening, config)
                    } else {
                        play_game(second.as_mut(), first.as_mut(), &job.opening, config)
                    };
                    if tx.send((i, record)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        for (i, record) in rx {
            on_game(i, &jobs[i], record);
        }
    });
}

/// A small, seedable random number generator (SplitMix64). Plenty for picking
/// opening moves, and the same seed always gives the same openings.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
