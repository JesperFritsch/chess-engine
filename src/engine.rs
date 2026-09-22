
mod eval;
mod search;
mod tt;
mod interface;

use shakmaty::Chess;
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
    ChessEngine
};


