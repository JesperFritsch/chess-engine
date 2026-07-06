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

pub struct Tt {
    pub entries: Vec<Option<TtEntry>>,
    pub mask: usize,
}

impl Tt {
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<Option<TtEntry>>();
        let num_entries = (size_mb * 1024 * 1024 / entry_size).next_power_of_two();
        Tt {
            entries: vec![None; num_entries],
            mask: num_entries - 1,
        } 
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    pub fn probe(&self, key: u64) -> Option<&TtEntry> {
        let e = self.entries[self.index(key)].as_ref();
        if let Some(e) = e {
            if e.key == key {
                return Some(e);
            }
        }
        None
    }

    pub fn store (&mut self, key: u64, depth: u32, score: i32, bound: Bound, best_move: Option<Move>) {
        let index = self.index(key);
        if let Some(e) = &self.entries[index] {
            if e.key == key && e.depth > depth {
                return; // Don't overwrite a deeper entry with a shallower one
            }
        }
        let entry = TtEntry { key, depth, score, bound, best_move };
        self.entries[index] = Some(entry);
    }

}