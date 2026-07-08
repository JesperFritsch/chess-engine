use chess_engine::engine::{depth_bound_search, SearchContext};
use shakmaty::{CastlingMode, Chess};
use shakmaty::fen::Fen;
use std::time::Instant;

/// Run explicitly with:
///   cargo test --release --test nodes_bench -- --ignored --nocapture
/// Used to A/B the impact of each search improvement.
#[test]
#[ignore = "manual benchmark; run with --ignored --nocapture"]
fn bench_nodes_to_fixed_depth() {
    // "Kiwipete" — a busy middlegame position used for perft/search benchmarks.
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let pos: Chess = format!("{fen}")
        .parse::<Fen>()
        .unwrap()
        .into_position(CastlingMode::Standard)
        .unwrap();

    // How deep can we get in a fixed 2-second budget? This is the metric that
    // maps to playing strength. The 2s deadline lives in the context, so an
    // iteration that runs out of time returns None and we stop.
    let mut ctx = SearchContext::new(256, 2000);
    let t = Instant::now();
    let mut reached = 0;
    let mut score = 0;
    for depth in 1..=64 {
        match depth_bound_search(&mut ctx, &pos, depth) {
            Some(s) => {
                score = s;
                reached = depth;
            }
            None => break, // deadline hit mid-iteration
        }
    }
    let dt = t.elapsed();
    println!(
        "in ~2s: reached depth {reached}, nodes={} score={score} time={:.3}s",
        ctx.node_count,
        dt.as_secs_f64()
    );
}
