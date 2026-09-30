use chess_engine::engine;
use chess_engine::engine::{ChessEngine};
use shakmaty::{CastlingMode, Chess};
use shakmaty::fen::Fen;
use std::time::{Instant};
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

    let t = Instant::now();
    let mut engine = engine::SearchEngine::new(500, 1);
    engine.search_handle().set_limits(
        engine::Limits {
            time_mode: engine::TimeMode::Unbound,
            max_depth: Some(5),
            max_nodes: None,
            restrict_to: None
        }
    );
    engine.set_position(pos.clone(), &[]);
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

// Expected print:
//
// chess-engine main  ❯ cargo test --test nodes_bench -- --nocapture --ignored
//     Finished `test` profile [unoptimized + debuginfo] target(s) in 0.02s
//      Running tests/nodes_bench.rs (target/debug/deps/nodes_bench-665e565de794bf36)
//
// running 1 test
// in ~2s: reached depth 5, nodes=64815 score=+0.40 time=0.512s
// test bench_nodes_to_fixed_depth ... ok
//
// test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.51s
