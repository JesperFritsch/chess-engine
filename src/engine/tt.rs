use shakmaty::Move;


#[derive(Clone, Copy, PartialEq)]
pub enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
pub struct TtEntry {
    pub key: u64,
    pub depth: u32,
    pub score: i32,
    pub bound: Bound,
    pub best_move: Option<Move>,
}

impl TtEntry {
    /// An unused slot. `key == 0` is the "empty" sentinel, which keeps every
    /// slot a plain `TtEntry` instead of an `Option<TtEntry>` (no discriminant,
    /// no padding, more entries per MiB). The one position in 2^64 that really
    /// hashes to 0 is simply never cached — `probe` and `store` both skip it.
    const EMPTY: TtEntry = TtEntry {
        key: 0,
        depth: 0,
        score: 0,
        bound: Bound::Exact,
        best_move: None,
    };

    fn is_empty(&self) -> bool {
        self.key == 0
    }
}

/// Never shrink below this, so a tiny `Hash` value still leaves a usable table.
const MIN_ENTRIES: usize = 1024;

/// How many entries fit in `size_mb`, rounded *down* to a power of two.
///
/// Rounding down matters: the index is `key & mask`, which needs a power of
/// two, and rounding up would let a requested 64 MiB allocate nearly 128 MiB —
/// a GUI setting `Hash` expects the limit to be respected.
fn entry_count(size_mb: usize) -> usize {
    let fits = size_mb * 1024 * 1024 / std::mem::size_of::<TtEntry>();
    let rounded = if fits.is_power_of_two() {
        fits
    } else {
        fits.next_power_of_two() >> 1
    };
    rounded.max(MIN_ENTRIES)
}

pub struct Tt {
    entries: Vec<TtEntry>,
    mask: usize,
    /// Occupied slots, maintained by `store` so `hashfull` is O(1).
    filled: usize,
}

impl Tt {
    pub fn new(size_mb: usize) -> Self {
        let num_entries = entry_count(size_mb);
        Tt {
            entries: vec![TtEntry::EMPTY; num_entries],
            mask: num_entries - 1,
            filled: 0,
        }
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    pub fn probe(&self, key: u64) -> Option<&TtEntry> {
        if key == 0 {
            return None; // Would match every untouched slot.
        }
        let e = &self.entries[self.index(key)];
        if e.key == key {
            return Some(e);
        }
        None
    }

    pub fn store(&mut self, key: u64, depth: u32, score: i32, bound: Bound, best_move: Option<Move>) {
        if key == 0 {
            return; // Can't be distinguished from an empty slot.
        }
        let index = self.index(key);
        let slot = &mut self.entries[index];
        if slot.key == key && slot.depth > depth {
            return; // Don't overwrite a deeper entry with a shallower one
        }
        if slot.is_empty() {
            self.filled += 1;
        }
        *slot = TtEntry { key, depth, score, bound, best_move };
    }

    /// Drop every entry, keeping the current size. Used between games, where
    /// entries from the previous game are worse than useless.
    pub fn clear(&mut self) {
        self.entries.fill(TtEntry::EMPTY);
        self.filled = 0;
    }

    pub fn resize(&mut self, size_mb: usize) {
        let num_entries = entry_count(size_mb);
        self.entries = vec![TtEntry::EMPTY; num_entries];
        self.mask = num_entries - 1;
        self.filled = 0;
    }

    pub fn fill_fraction(&self) -> f32 {
        (self.filled as f32 / self.entries.len() as f32) as f32
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_and_probes() {
        let mut tt = Tt::new(1);
        tt.store(0xDEAD_BEEF, 5, 42, Bound::Exact, None);
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
    fn key_zero_is_never_cached() {
        let mut tt = Tt::new(1);
        tt.store(0, 9, 123, Bound::Exact, None);
        assert!(tt.probe(0).is_none(), "key 0 must not match empty slots");
        assert_eq!(tt.fill_fraction(), 0.0);
    }

    #[test]
    fn a_deeper_entry_survives_a_shallower_store() {
        let mut tt = Tt::new(1);
        tt.store(7, 8, 100, Bound::Exact, None);
        tt.store(7, 3, 200, Bound::Exact, None);
        assert_eq!(tt.probe(7).unwrap().score, 100);
    }

    #[test]
    fn filled_counts_slots_not_stores() {
        let mut tt = Tt::new(1);
        tt.store(7, 1, 0, Bound::Exact, None);
        tt.store(7, 2, 0, Bound::Exact, None); // same slot, replaced
        assert_eq!(tt.filled, 1);
    }

    #[test]
    fn clear_empties_the_table() {
        let mut tt = Tt::new(1);
        tt.store(7, 1, 0, Bound::Exact, None);
        tt.clear();
        assert_eq!(tt.fill_fraction(), 0.0);
        assert!(tt.probe(7).is_none());
    }

    #[test]
    fn size_never_exceeds_the_request() {
        for mb in [1usize, 2, 3, 7, 64, 100] {
            let tt = Tt::new(mb);
            let bytes = tt.entries.len() * std::mem::size_of::<TtEntry>();
            assert!(
                bytes <= mb * 1024 * 1024 || tt.entries.len() == MIN_ENTRIES,
                "{mb} MiB request allocated {bytes} bytes"
            );
            assert!(tt.entries.len().is_power_of_two());
        }
    }
}
