
mod eval;
mod search;
mod tt;
mod interface;

pub use tt::{
    Tt,
    TtEntry,
};

pub use search::{
    depth_bound_search,
    mate_in,
    time_bound_search,
    time_bound_search_with_progress,
    SearchContext,
    SearchInfo,
};


pub use interface::{
    SearchControl,
    SearchHandle,
    SearchProgress,
    SearchResult,
    ChessEngine
};
