use chess_engine::engine;
use chess_engine::engine::{ChessEngine, Score};
use shakmaty::{CastlingMode, Chess};
use shakmaty::fen::Fen;
use std::time::{Instant, Duration};
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
    let t = Instant::now();
    let mut engine = engine::SearchEngine::new(500);
    engine.search_handle().set_limits(
        engine::Limits {
            time_mode: engine::TimeMode::Fixed(Duration::from_secs_f64(2.0)),
            max_depth: None,
            max_nodes: None,
            restrict_to: None
        }
    );
    engine.set_position(pos.clone());
    let res = engine.search(&mut |_|{});

    let dt = t.elapsed();
    println!(
        "in ~2s: reached depth {}, nodes={} score={} time={:.3}s",
        res.depth,
        res.nodes,
        res.score.unwrap(),
        dt.as_secs_f64()
    );
}
