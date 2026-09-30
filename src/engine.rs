
mod eval;
mod search;
mod tt;
mod interface;

pub use tt::{
    Tt,
    TtEntry,
};

pub use search::{
    SearchStats,
    mate_in,
    SearchEngine
};


pub use interface::{
    SearchHandle,
    SearchProgress,
    SearchResult,
    ChessEngine,
    Limits,
    IllegalMove,
    Score,
    TimeMode,
    Clock,
};


