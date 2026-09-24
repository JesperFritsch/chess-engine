//! A terminal front end for playing against the engine.
//!
//! This is one concrete implementation of [`crate::play::Frontend`]. It owns
//! the terminal (raw mode + alternate screen) and drives an interactive,
//! keyboard-only move picker:
//!
//! * arrow keys move a highlighted cursor square around the board,
//! * `Enter` on one of your pieces selects it and lights up its legal targets,
//! * `Enter` on a highlighted target plays that move,
//! * `Backspace` cancels the current selection,
//! * `q` / `Esc` quits the game.
//!
//! The whole screen is cleared and redrawn on every change, so the terminal
//! never accumulates a scrollback of old positions.

use std::io::{self, Stdout, Write};

use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute, queue,
    style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor},
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use shakmaty::{Color as ChessColor, File, Move, Position, Rank, Role, Square};

use crate::engine::{SearchProgress};
use crate::play::{BoardView, Frontend, PlayerMove};

// --- Palette -----------------------------------------------------------------

const LIGHT_SQUARE: Color = Color::Rgb { r: 240, g: 217, b: 181 };
const DARK_SQUARE: Color = Color::Rgb { r: 181, g: 136, b: 99 };
const SELECTED: Color = Color::Rgb { r: 246, g: 231, b: 92 };
const TARGET: Color = Color::Rgb { r: 124, g: 172, b: 96 };
const LAST_MOVE: Color = Color::Rgb { r: 206, g: 210, b: 107 };
const WHITE_PIECE: Color = Color::Rgb { r: 250, g: 250, b: 250 };
const BLACK_PIECE: Color = Color::Rgb { r: 30, g: 30, b: 30 };
const LABEL: Color = Color::Rgb { r: 140, g: 140, b: 140 };

/// Amount each RGB channel is brightened for the square under the cursor, so
/// the cursor reads as "the same square, lit up" regardless of what's beneath.
const CURSOR_LIFT: u8 = 45;

// --- Front end ---------------------------------------------------------------

pub struct TuiFrontend {
    out: Stdout,
}

impl TuiFrontend {
    /// Take over the terminal: enable raw mode and switch to the alternate
    /// screen. The terminal is restored when the value is dropped.
    pub fn new() -> io::Result<Self> {
        let mut out = io::stdout();
        terminal::enable_raw_mode()?;
        execute!(out, EnterAlternateScreen, cursor::Hide)?;
        Ok(TuiFrontend { out })
    }
}

impl Drop for TuiFrontend {
    fn drop(&mut self) {
        // Best-effort restore; nothing useful to do if this fails while unwinding.
        let _ = execute!(self.out, cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

impl Frontend for TuiFrontend {
    fn choose_side(&mut self) -> io::Result<Option<ChessColor>> {
        // A tiny two-option picker in the same spirit as the move picker.
        let mut choice = 0usize; // 0 = White, 1 = Black
        loop {
            self.draw_side_menu(choice)?;
            match next_key(self)? {
                Some(KeyCode::Left) | Some(KeyCode::Up) => choice = 0,
                Some(KeyCode::Right) | Some(KeyCode::Down) => choice = 1,
                Some(KeyCode::Char('w')) => return Ok(Some(ChessColor::White)),
                Some(KeyCode::Char('b')) => return Ok(Some(ChessColor::Black)),
                Some(KeyCode::Enter) => {
                    return Ok(Some(if choice == 0 {
                        ChessColor::White
                    } else {
                        ChessColor::Black
                    }));
                }
                Some(KeyCode::Char('q')) | Some(KeyCode::Esc) => return Ok(None),
                _ => {}
            }
        }
    }

    fn request_move(&mut self, view: &BoardView) -> io::Result<PlayerMove> {
        let legal: Vec<Move> = view.pos.legal_moves().iter().copied().collect();

        // Start the cursor on a piece we can actually move.
        let mut cursor = legal
            .first()
            .and_then(|m| m.from())
            .unwrap_or_else(|| Square::from_coords(File::new(4), Rank::new(1)));
        let mut selected: Option<Square> = None;
        let mut targets: Vec<Square> = Vec::new();

        loop {
            self.draw_board(view, Some(cursor), selected, &targets)?;

            let Some(code) = next_key(self)? else { continue };
            match code {
                KeyCode::Left => cursor = step(cursor, view.perspective, -1, 0),
                KeyCode::Right => cursor = step(cursor, view.perspective, 1, 0),
                KeyCode::Up => cursor = step(cursor, view.perspective, 0, 1),
                KeyCode::Down => cursor = step(cursor, view.perspective, 0, -1),
                KeyCode::Backspace => {
                    selected = None;
                    targets.clear();
                }
                KeyCode::Char('q') | KeyCode::Esc => return Ok(PlayerMove::Quit),
                KeyCode::Enter => {
                    match selected {
                        // A piece is selected: is the cursor on a legal target?
                        Some(from) if targets.contains(&cursor) => {
                            let mv = self.pick_move(view, &legal, from, cursor)?;
                            return Ok(PlayerMove::Play(mv));
                        }
                        // Otherwise treat Enter as (re)selecting whatever is under
                        // the cursor, if it's a movable piece.
                        _ => {
                            let sq_targets = targets_from(&legal, cursor);
                            if !sq_targets.is_empty() {
                                selected = Some(cursor);
                                targets = sq_targets;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn show(&mut self, view: &BoardView) -> io::Result<()> {
        self.draw_board(view, None, None, &[])
    }

    fn thinking(&mut self, _view: &BoardView, info: &SearchProgress) {
        // Best-effort live update. The board doesn't change while the engine
        // thinks, so we only refresh the info line beneath it (drawn once by the
        // preceding `show`). Ignore write errors — this is cosmetic.
        let _ = self.draw_thinking(info);
    }

    fn game_over(&mut self, view: &BoardView, pgn: &str) -> io::Result<()> {
        self.draw_board(view, None, None, &[])?;
        self.footer("Press s to save this game as PGN, or any other key to quit")?;
        loop {
            match next_key(self)? {
                Some(KeyCode::Char('s')) | Some(KeyCode::Char('S')) => {
                    self.save_pgn(view, pgn)?;
                    return Ok(());
                }
                Some(_) => return Ok(()),
                None => {}
            }
        }
    }
}

impl TuiFrontend {
    /// Redraw the whole screen: title, board (oriented for the human), file/rank
    /// labels, and the status line.
    fn draw_board(
        &mut self,
        view: &BoardView,
        cursor: Option<Square>,
        selected: Option<Square>,
        targets: &[Square],
    ) -> io::Result<()> {
        let flip = view.perspective == ChessColor::Black;
        let w = &mut self.out;

        queue!(w, Clear(ClearType::All), cursor::MoveTo(0, 0), ResetColor)?;
        queue!(w, SetForegroundColor(LABEL), Print("  Rust Chess"))?;

        for vrow in 0..8u32 {
            let y = (vrow + 2) as u16;
            let rank_idx = if flip { vrow } else { 7 - vrow };
            queue!(
                w,
                cursor::MoveTo(0, y),
                ResetColor,
                SetForegroundColor(LABEL),
                Print(format!(" {} ", rank_idx + 1))
            )?;

            for vcol in 0..8u32 {
                let file_idx = if flip { 7 - vcol } else { vcol };
                let sq = Square::from_coords(File::new(file_idx), Rank::new(rank_idx));
                let bg = square_bg(sq, view, cursor, selected, targets);
                let (glyph, fg) = match view.pos.board().piece_at(sq) {
                    Some(piece) => (
                        piece_glyph(piece.role),
                        if piece.color == ChessColor::White {
                            WHITE_PIECE
                        } else {
                            BLACK_PIECE
                        },
                    ),
                    None => (' ', WHITE_PIECE),
                };
                queue!(
                    w,
                    SetBackgroundColor(bg),
                    SetForegroundColor(fg),
                    Print(format!(" {glyph} "))
                )?;
            }
            queue!(w, ResetColor)?;
        }

        // File labels along the bottom.
        let mut files = String::from("   ");
        for vcol in 0..8u32 {
            let file_idx = if flip { 7 - vcol } else { vcol };
            files.push(' ');
            files.push(File::new(file_idx).char());
            files.push(' ');
        }
        queue!(
            w,
            cursor::MoveTo(0, 10),
            ResetColor,
            SetForegroundColor(LABEL),
            Print(files)
        )?;

        // Status / prompt line.
        queue!(
            w,
            cursor::MoveTo(0, 12),
            ResetColor,
            Print(&view.status)
        )?;
        queue!(
            w,
            cursor::MoveTo(0, 14),
            SetForegroundColor(LABEL),
            Print("↑↓←→ move   Enter select/confirm   Backspace cancel   q quit"),
            ResetColor
        )?;

        w.flush()
    }

    /// Refresh the live "engine thinking" info line (depth / score / speed)
    /// without redrawing the static board. The principal variation is
    /// deliberately *not* shown — it would reveal the engine's expected best line
    /// (including the human's replies), an unfair hint.
    fn draw_thinking(&mut self, info: &SearchProgress) -> io::Result<()> {
        let secs = info.elapsed.as_secs_f64();
        let knps = if secs > 0.0 { info.nodes as f64 / secs / 1000.0 } else { 0.0 };
        let line = format!(
            "Engine thinking…  depth {}  score {}  {:.0}kn  {:.0}kn/s  {:.1}s",
            info.depth,
            &info.score,
            info.nodes as f64 / 1000.0,
            knps,
            secs,
        );

        let w = &mut self.out;
        queue!(
            w,
            cursor::MoveTo(0, 12),
            Clear(ClearType::CurrentLine),
            ResetColor,
            Print(line)
        )?;
        w.flush()
    }

    /// Overwrite the footer/help line with a one-off message.
    fn footer(&mut self, text: &str) -> io::Result<()> {
        queue!(
            self.out,
            cursor::MoveTo(0, 14),
            Clear(ClearType::CurrentLine),
            ResetColor,
            SetForegroundColor(LABEL),
            Print(text),
            ResetColor
        )?;
        self.out.flush()
    }

    /// Prompt (in the footer) for a line of text, pre-filled with `default`.
    /// Returns `Ok(None)` if the user pressed Esc to cancel.
    fn read_line(
        &mut self,
        view: &BoardView,
        prompt: &str,
        default: &str,
    ) -> io::Result<Option<String>> {
        let mut buf = String::from(default);
        loop {
            self.draw_board(view, None, None, &[])?;
            self.footer(&format!(
                "{prompt}: {buf}\u{2588}   (Enter to confirm, Esc to cancel)"
            ))?;
            match next_key(self)? {
                Some(KeyCode::Enter) => return Ok(Some(buf)),
                Some(KeyCode::Esc) => return Ok(None),
                Some(KeyCode::Backspace) => {
                    buf.pop();
                }
                Some(KeyCode::Char(c)) => buf.push(c),
                _ => {}
            }
        }
    }

    /// Ask for a filename and write the PGN, reporting the outcome.
    fn save_pgn(&mut self, view: &BoardView, pgn: &str) -> io::Result<()> {
        let name = match self.read_line(view, "Save as", "game.pgn")? {
            Some(name) => name,
            None => return Ok(()), // cancelled
        };
        let name = name.trim();
        let path = if name.is_empty() { "game.pgn" } else { name };

        self.draw_board(view, None, None, &[])?;
        match std::fs::write(path, pgn) {
            Ok(()) => self.footer(&format!("Saved to {path} — press any key to quit"))?,
            Err(err) => self.footer(&format!("Could not save: {err} — press any key to quit"))?,
        }
        loop {
            if next_key(self)?.is_some() {
                return Ok(());
            }
        }
    }

    fn draw_side_menu(&mut self, choice: usize) -> io::Result<()> {
        let w = &mut self.out;
        queue!(w, Clear(ClearType::All), cursor::MoveTo(0, 0), ResetColor)?;
        queue!(
            w,
            cursor::MoveTo(2, 1),
            SetForegroundColor(LABEL),
            Print("Rust Chess"),
            ResetColor
        )?;
        queue!(w, cursor::MoveTo(2, 3), Print("Choose your side:"))?;

        let options = ["  White  ", "  Black  "];
        for (i, label) in options.iter().enumerate() {
            let x = 2 + (i as u16) * 12;
            if i == choice {
                queue!(
                    w,
                    cursor::MoveTo(x, 5),
                    SetBackgroundColor(SELECTED),
                    SetForegroundColor(BLACK_PIECE),
                    Print(*label),
                    ResetColor
                )?;
            } else {
                queue!(
                    w,
                    cursor::MoveTo(x, 5),
                    SetBackgroundColor(DARK_SQUARE),
                    SetForegroundColor(WHITE_PIECE),
                    Print(*label),
                    ResetColor
                )?;
            }
        }
        queue!(
            w,
            cursor::MoveTo(2, 7),
            SetForegroundColor(LABEL),
            Print("←→ choose   Enter confirm   (or press w / b)   q quit"),
            ResetColor
        )?;
        w.flush()
    }

    /// Resolve the concrete move from `from` to `to`. If it's a promotion there
    /// will be several candidates (one per promotion piece); ask which.
    fn pick_move(
        &mut self,
        view: &BoardView,
        legal: &[Move],
        from: Square,
        to: Square,
    ) -> io::Result<Move> {
        let candidates: Vec<Move> = legal
            .iter()
            .copied()
            .filter(|m| m.from() == Some(from) && m.to() == to)
            .collect();

        match candidates.as_slice() {
            [] => unreachable!("pick_move called for a non-target square"),
            [only] => Ok(*only),
            _ => self.pick_promotion(view, &candidates),
        }
    }

    fn pick_promotion(&mut self, view: &BoardView, candidates: &[Move]) -> io::Result<Move> {
        // Present the promotion pieces in a stable order.
        let order = [Role::Queen, Role::Rook, Role::Bishop, Role::Knight];
        let mut choices: Vec<Move> = Vec::new();
        for role in order {
            if let Some(m) = candidates.iter().find(|m| m.promotion() == Some(role)) {
                choices.push(*m);
            }
        }
        // Fallback: if for some reason nothing matched, just take the first.
        if choices.is_empty() {
            return Ok(candidates[0]);
        }

        let mut idx = 0usize;
        loop {
            self.draw_promotion(view, &choices, idx)?;
            match next_key(self)? {
                Some(KeyCode::Left) | Some(KeyCode::Up) => {
                    idx = (idx + choices.len() - 1) % choices.len();
                }
                Some(KeyCode::Right) | Some(KeyCode::Down) => {
                    idx = (idx + 1) % choices.len();
                }
                Some(KeyCode::Enter) => return Ok(choices[idx]),
                Some(KeyCode::Char(c)) => {
                    if let Some(role) = role_from_char(c)
                        && let Some(pos) =
                            choices.iter().position(|m| m.promotion() == Some(role))
                    {
                        return Ok(choices[pos]);
                    }
                }
                _ => {}
            }
        }
    }

    fn draw_promotion(&mut self, view: &BoardView, choices: &[Move], idx: usize) -> io::Result<()> {
        self.draw_board(view, None, None, &[])?;
        let w = &mut self.out;
        queue!(
            w,
            cursor::MoveTo(0, 12),
            ResetColor,
            Print("Promote to:  ")
        )?;
        for (i, m) in choices.iter().enumerate() {
            let role = m.promotion().unwrap_or(Role::Queen);
            let label = format!(" {} ", role.upper_char());
            if i == idx {
                queue!(
                    w,
                    SetBackgroundColor(SELECTED),
                    SetForegroundColor(BLACK_PIECE),
                    Print(label),
                    ResetColor,
                    Print(" ")
                )?;
            } else {
                queue!(
                    w,
                    SetBackgroundColor(DARK_SQUARE),
                    SetForegroundColor(WHITE_PIECE),
                    Print(label),
                    ResetColor,
                    Print(" ")
                )?;
            }
        }
        w.flush()
    }
}

// --- Helpers -----------------------------------------------------------------

/// The Unicode chess glyph for a role. We always use the *solid* figures
/// (`U+265A`–`U+265F`) for both colors and let the foreground color say which
/// side owns the piece — the outline (white) glyphs render inconsistently and
/// look faint on colored squares.
fn piece_glyph(role: Role) -> char {
    match role {
        Role::King => '\u{265A}',   // ♚
        Role::Queen => '\u{265B}',  // ♛
        Role::Rook => '\u{265C}',   // ♜
        Role::Bishop => '\u{265D}', // ♝
        Role::Knight => '\u{265E}', // ♞
        Role::Pawn => '\u{265F}',   // ♟
    }
}


/// Legal destination squares for a piece standing on `from`.
fn targets_from(legal: &[Move], from: Square) -> Vec<Square> {
    let mut targets: Vec<Square> = legal
        .iter()
        .filter(|m| m.from() == Some(from))
        .map(|m| m.to())
        .collect();
    targets.sort_by_key(|s| s.to_u32());
    targets.dedup();
    targets
}

/// Move the cursor by `(dcol, drow)` in *screen* space (right/up positive),
/// respecting the board orientation and clamping at the edges.
fn step(sq: Square, perspective: ChessColor, dcol: i32, drow: i32) -> Square {
    let flip = perspective == ChessColor::Black;
    let file = sq.file().to_u32() as i32;
    let rank = sq.rank().to_u32() as i32;

    // Screen-right increases file for White, decreases it for Black; screen-up
    // increases rank for White, decreases it for Black.
    let (nf, nr) = if flip {
        (file - dcol, rank - drow)
    } else {
        (file + dcol, rank + drow)
    };

    let nf = nf.clamp(0, 7) as u32;
    let nr = nr.clamp(0, 7) as u32;
    Square::from_coords(File::new(nf), Rank::new(nr))
}

/// Pick the background color for a square, layering interaction state over the
/// checkerboard and lifting brightness for the cursor square.
fn square_bg(
    sq: Square,
    view: &BoardView,
    cursor: Option<Square>,
    selected: Option<Square>,
    targets: &[Square],
) -> Color {
    let base = if selected == Some(sq) {
        SELECTED
    } else if targets.contains(&sq) {
        TARGET
    } else if view.last_move.is_some_and(|m| m.from() == Some(sq) || m.to() == sq) {
        LAST_MOVE
    } else if (sq.file().to_u32() + sq.rank().to_u32()).is_multiple_of(2) {
        DARK_SQUARE
    } else {
        LIGHT_SQUARE
    };

    if cursor == Some(sq) {
        brighten(base, CURSOR_LIFT)
    } else {
        base
    }
}

fn brighten(color: Color, amount: u8) -> Color {
    match color {
        Color::Rgb { r, g, b } => Color::Rgb {
            r: r.saturating_add(amount),
            g: g.saturating_add(amount),
            b: b.saturating_add(amount),
        },
        other => other,
    }
}

fn role_from_char(c: char) -> Option<Role> {
    match c.to_ascii_lowercase() {
        'q' => Some(Role::Queen),
        'r' => Some(Role::Rook),
        'b' => Some(Role::Bishop),
        'n' => Some(Role::Knight),
        _ => None,
    }
}

/// Block for the next key *press*, translating Ctrl-C into a quit. Returns
/// `Ok(None)` for events we don't care about (resize, key release, …) so the
/// caller can loop.
#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::Chess;

    fn sq(file: u32, rank: u32) -> Square {
        Square::from_coords(File::new(file), Rank::new(rank))
    }

    #[test]
    fn cursor_moves_in_screen_space_for_white() {
        let e2 = sq(4, 1);
        // Screen-up advances the pawn's rank; screen-right moves toward the h-file.
        assert_eq!(step(e2, ChessColor::White, 0, 1), sq(4, 2)); // e3
        assert_eq!(step(e2, ChessColor::White, 1, 0), sq(5, 1)); // f2
        assert_eq!(step(e2, ChessColor::White, -1, 0), sq(3, 1)); // d2
    }

    #[test]
    fn cursor_is_flipped_for_black() {
        let e7 = sq(4, 6);
        // From Black's perspective, screen-up is toward Black's own advance.
        assert_eq!(step(e7, ChessColor::Black, 0, 1), sq(4, 5)); // e6
        assert_eq!(step(e7, ChessColor::Black, 1, 0), sq(3, 6)); // d7 (screen-right)
    }

    #[test]
    fn cursor_clamps_at_edges() {
        let a1 = sq(0, 0);
        assert_eq!(step(a1, ChessColor::White, -1, -1), a1);
        let h8 = sq(7, 7);
        assert_eq!(step(h8, ChessColor::White, 1, 1), h8);
    }

    #[test]
    fn targets_from_start_position() {
        let pos = Chess::default();
        let legal: Vec<Move> = pos.legal_moves().iter().copied().collect();

        // The knight on b1 can reach a3 and c3 only.
        let mut knight = targets_from(&legal, sq(1, 0));
        knight.sort_by_key(|s| s.to_u32());
        assert_eq!(knight, vec![sq(0, 2), sq(2, 2)]);

        // The e2 pawn can push to e3 and e4.
        let mut pawn = targets_from(&legal, sq(4, 1));
        pawn.sort_by_key(|s| s.to_u32());
        assert_eq!(pawn, vec![sq(4, 2), sq(4, 3)]);

        // An empty square yields nothing selectable.
        assert!(targets_from(&legal, sq(4, 3)).is_empty());
    }

    #[test]
    fn cursor_lifts_brightness_over_any_base() {
        let view = BoardView {
            pos: Chess::default(),
            perspective: ChessColor::White,
            last_move: None,
            status: String::new(),
        };
        let target = sq(3, 3);
        let plain = square_bg(target, &view, None, None, &[target]);
        let lit = square_bg(target, &view, Some(target), None, &[target]);
        assert_ne!(plain, lit, "cursor square should be visibly brighter");
    }
}

fn next_key(front: &mut TuiFrontend) -> io::Result<Option<KeyCode>> {
    match event::read()? {
        Event::Key(KeyEvent {
            code,
            kind,
            modifiers,
            ..
        }) => {
            if kind == KeyEventKind::Release {
                return Ok(None);
            }
            if modifiers.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
                // `process::exit` skips destructors, so restore the terminal by
                // hand before bailing out hard.
                let _ = execute!(front.out, cursor::Show, LeaveAlternateScreen);
                let _ = terminal::disable_raw_mode();
                std::process::exit(130);
            }
            Ok(Some(code))
        }
        _ => Ok(None),
    }
}
