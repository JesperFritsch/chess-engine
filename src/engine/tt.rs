use shakmaty::{Move, Role};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

#[derive(Clone, Copy, PartialEq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

/// A move squeezed into 16 bits as `from(6) | to(6) | promotion(3)`.
///
/// Zero means "no move": a1→a1 is not a legal move, so the value is free.
/// The encoding is lossy — it does not record whether the move was a capture,
/// en passant or a castle — so it is never turned back into a [`Move`].
/// Callers compare `pack_move(candidate) == packed` against the moves they have
/// generated, which both resolves the move and validates it.
pub type PackedMove = u16;

pub fn pack_move(m: Move) -> PackedMove {
    let from = m.from().map_or(0, |s| s.to_usize()) as u16;
    let to = m.to().to_usize() as u16;
    let promotion = match m.promotion() {
        Some(Role::Knight) => 1,
        Some(Role::Bishop) => 2,
        Some(Role::Rook) => 3,
        Some(Role::Queen) => 4,
        _ => 0,
    };
    from | (to << 6) | (promotion << 12)
}

/// One slot, unpacked. Never stored in this form — see [`pack`].
#[derive(Clone, Copy)]
pub struct TtEntry {
    pub depth: u8,
    pub score: i16,
    pub bound: Bound,
    pub mv: PackedMove,
}

// A slot is a single u64 so that one atomic load or store covers the whole
// entry: no locking, and no way for a reader to see half of one write and half
// of another.
//
//   bits  0..16  score, i16
//   bits 16..24  depth, u8 — 0 marks an unused slot
//   bits 24..26  bound
//   bits 26..42  packed move
//   bits 42..64  the top 22 bits of the key
//
// Only 22 bits of the key survive. Together with the index bits that selected
// the slot that is enough to make a wrong hit rare, and a wrong hit is bounded:
// the move is re-validated against the legal moves, and the score still has to
// satisfy the depth and bound tests to be used at all.
const DEPTH_SHIFT: u32 = 16;
const BOUND_SHIFT: u32 = 24;
const MOVE_SHIFT: u32 = 26;
const SIG_SHIFT: u32 = 42;

fn signature(key: u64) -> u64 {
    key >> SIG_SHIFT
}

fn pack(key: u64, depth: u8, score: i16, bound: Bound, mv: PackedMove) -> u64 {
    (score as u16 as u64)
        | (depth as u64) << DEPTH_SHIFT
        | (bound as u64) << BOUND_SHIFT
        | (mv as u64) << MOVE_SHIFT
        | signature(key) << SIG_SHIFT
}

fn unpack(word: u64) -> TtEntry {
    TtEntry {
        score: (word & 0xFFFF) as u16 as i16,
        depth: depth_of(word),
        bound: match (word >> BOUND_SHIFT) & 0b11 {
            1 => Bound::Lower,
            2 => Bound::Upper,
            _ => Bound::Exact,
        },
        mv: ((word >> MOVE_SHIFT) & 0xFFFF) as PackedMove,
    }
}

fn depth_of(word: u64) -> u8 {
    ((word >> DEPTH_SHIFT) & 0xFF) as u8
}

/// Never shrink below this, so a tiny `Hash` value still leaves a usable table.
const MIN_ENTRIES: usize = 1024;

fn entry_count(size_mb: usize) -> usize {
    let fits = size_mb * 1024 * 1024 / std::mem::size_of::<AtomicU64>();
    let rounded = if fits.is_power_of_two() {
        fits
    } else {
        fits.next_power_of_two() >> 1
    };
    rounded.max(MIN_ENTRIES)
}

fn zeroed(num_entries: usize) -> Vec<AtomicU64> {
    let mut entries = Vec::with_capacity(num_entries);
    entries.resize_with(num_entries, || AtomicU64::new(0));
    entries
}

pub struct Tt {
    entries: Vec<AtomicU64>,
    mask: usize,
    /// Occupied slots, maintained by `store` so `hashfull` is O(1). Approximate
    /// once several threads store concurrently, which is all it needs to be.
    filled: AtomicUsize,
}

impl Tt {
    pub fn new(size_mb: usize) -> Self {
        let num_entries = entry_count(size_mb);
        Tt {
            entries: zeroed(num_entries),
            mask: num_entries - 1,
            filled: AtomicUsize::new(0),
        }
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    pub fn probe(&self, key: u64) -> Option<TtEntry> {
        let word = self.entries[self.index(key)].load(Relaxed);
        if depth_of(word) == 0 || word >> SIG_SHIFT != signature(key) {
            return None;
        }
        Some(unpack(word))
    }

    pub fn store(&self, key: u64, depth: u8, score: i16, bound: Bound, mv: PackedMove) {
        debug_assert!(depth > 0, "depth 0 is the empty-slot marker");
        let slot = &self.entries[self.index(key)];
        let prev = slot.load(Relaxed);
        if prev >> SIG_SHIFT == signature(key) && depth_of(prev) > depth {
            return; // Don't overwrite a deeper entry with a shallower one
        }
        if prev == 0 {
            self.filled.fetch_add(1, Relaxed);
        }
        slot.store(pack(key, depth, score, bound, mv), Relaxed);
    }

    /// Drop every entry, keeping the current size. Used between games, where
    /// entries from the previous game are worse than useless.
    pub fn clear(&mut self) {
        for slot in &self.entries {
            slot.store(0, Relaxed);
        }
        self.filled.store(0, Relaxed);
    }

    pub fn resize(&mut self, size_mb: usize) {
        let num_entries = entry_count(size_mb);
        self.entries = Vec::new(); // drop the old table before allocating the new one
        self.entries = zeroed(num_entries);
        self.mask = num_entries - 1;
        self.filled.store(0, Relaxed);
    }

    pub fn fill_fraction(&self) -> f32 {
        self.filled.load(Relaxed) as f32 / self.entries.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::{Chess, Position};

    fn first_legal(pos: &Chess) -> Move {
        pos.legal_moves()[0]
    }

    #[test]
    fn stores_and_probes() {
        let tt = Tt::new(1);
        tt.store(0xDEAD_BEEF, 5, 42, Bound::Exact, 0);
        let e = tt.probe(0xDEAD_BEEF).expect("entry should be found");
        assert_eq!(e.depth, 5);
        assert_eq!(e.score, 42);
    }

    #[test]
    fn a_miss_returns_none() {
        let tt = Tt::new(1);
        assert!(tt.probe(0xDEAD_BEEF).is_none());
    }

    #[test]
    fn every_field_survives_a_round_trip() {
        let tt = Tt::new(1);
        let mv = pack_move(first_legal(&Chess::default()));
        for (depth, score, bound) in [
            (1u8, -32001i16, Bound::Upper),
            (7, -1, Bound::Lower),
            (255, 32000, Bound::Exact),
        ] {
            tt.store(0x1234_5678_9ABC_DEF0, depth, score, bound, mv);
            let e = tt.probe(0x1234_5678_9ABC_DEF0).unwrap();
            assert_eq!(e.depth, depth);
            assert_eq!(e.score, score);
            assert!(e.bound == bound);
            assert_eq!(e.mv, mv);
        }
    }

    #[test]
    fn a_different_key_in_the_same_slot_is_a_miss() {
        let tt = Tt::new(1);
        // Same low bits (same slot), different high bits (different signature).
        let a = 0x0000_0000_0000_0007;
        let b = 0xFFFF_FC00_0000_0007;
        tt.store(a, 4, 11, Bound::Exact, 0);
        assert!(tt.probe(b).is_none());
        assert_eq!(tt.probe(a).unwrap().score, 11);
    }

    #[test]
    fn a_deeper_entry_survives_a_shallower_store() {
        let tt = Tt::new(1);
        tt.store(7, 8, 100, Bound::Exact, 0);
        tt.store(7, 3, 200, Bound::Exact, 0);
        assert_eq!(tt.probe(7).unwrap().score, 100);
    }

    #[test]
    fn filled_counts_slots_not_stores() {
        let tt = Tt::new(1);
        tt.store(7, 1, 0, Bound::Exact, 0);
        tt.store(7, 2, 0, Bound::Exact, 0); // same slot, replaced
        assert_eq!(tt.filled.load(Relaxed), 1);
    }

    #[test]
    fn clear_empties_the_table() {
        let mut tt = Tt::new(1);
        tt.store(7, 1, 0, Bound::Exact, 0);
        tt.clear();
        assert_eq!(tt.fill_fraction(), 0.0);
        assert!(tt.probe(7).is_none());
    }

    #[test]
    fn size_never_exceeds_the_request() {
        for mb in [1usize, 2, 3, 7, 64, 100] {
            let tt = Tt::new(mb);
            let bytes = tt.entries.len() * std::mem::size_of::<AtomicU64>();
            assert!(
                bytes <= mb * 1024 * 1024 || tt.entries.len() == MIN_ENTRIES,
                "{mb} MiB request allocated {bytes} bytes"
            );
            assert!(tt.entries.len().is_power_of_two());
        }
    }

    #[test]
    fn packed_moves_are_distinct_and_never_zero() {
        let pos = Chess::default();
        let mut seen = std::collections::HashSet::new();
        for m in pos.legal_moves() {
            let p = pack_move(m);
            assert_ne!(p, 0, "{m:?} packed to the no-move sentinel");
            assert!(seen.insert(p), "{m:?} collided with another move");
        }
    }
}
