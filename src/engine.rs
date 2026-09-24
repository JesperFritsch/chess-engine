
mod eval;
mod search;
mod tt;
mod interface;

pub use tt::{
    Tt,
    TtEntry,
};

pub use search::{
    mate_in,
    SearchContext,
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


