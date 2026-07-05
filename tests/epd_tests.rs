// tests/epd_tests.rs
use chess_engine::engine;
use chess_engine::render;
use shakmaty::{Chess, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty::san::San;
use std::fs;
use std::path::Path;

struct EpdCase {
    pos: Chess,
    best_move_san: String,
    id: String,
}

/// Parse an EPD line: "<4-field FEN> bm <move>; id \"...\";"
fn parse_epd(text: &str) -> EpdCase {
    let text = text.trim();

    // EPD FEN is 4 fields; append "0 1" to make a full 6-field FEN shakmaty accepts.
    let fields: Vec<&str> = text.split_whitespace().collect();
    let fen4 = fields[..4].join(" ");
    let full_fen = format!("{fen4} 0 1");

    let pos: Chess = full_fen
        .parse::<Fen>()
        .expect("valid FEN")
        .into_position(CastlingMode::Standard)
        .expect("legal position");

    // Extract the bm operation: everything between "bm " and the next ";"
    let bm = text
        .split("bm ")
        .nth(1)
        .and_then(|s| s.split(';').next())
        .expect("EPD must contain a bm operation")
        .trim()
        .to_string();

    // Extract id if present (optional, for nicer failure messages)
    let id = text
        .split("id ")
        .nth(1)
        .and_then(|s| s.split(';').next())
        .map(|s| s.trim().trim_matches('"').to_string())
        .unwrap_or_else(|| "unnamed".to_string());

    EpdCase { pos, best_move_san: bm, id }
}

/// Run the engine and check its best move equals the EPD's bm.
fn check_case(case: &EpdCase, renderer: &impl render::Renderer) {
    let result = engine::time_bound_best_line(&case.pos, 10000).unwrap(); // 1 second time limit
    let best = result.line.first().expect("engine returned no move");
    render::render_line(&case.pos, &result.line[0..], renderer);

    // Convert engine's move to SAN to compare against the EPD (which uses SAN).
    let played_san = San::from_move(&case.pos, *best).to_string();

    assert_eq!(
        played_san, case.best_move_san,
        "position '{}': engine played {}, expected {}",
        case.id, played_san, case.best_move_san
    );
}

/// Load and run every .epd file in tests/positions/.
#[test]
fn run_all_epd_positions() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/positions");
    let mut ran = 0;
    let renderer = render::TerminalRenderer;
    for entry in fs::read_dir(&dir).expect("positions dir exists") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("epd") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("readable epd file");
        let case = parse_epd(&text);
        check_case(&case, &renderer);   // depth 4; adjust per-file later if needed
        ran += 1;
    }

    assert!(ran > 0, "no .epd files found in {:?}", dir);
}

#[test]
#[ignore = "manual scenario runner; set SCENARIO=<file> and run explicitly"]
fn run_scenario() {
    let file = std::env::var("SCENARIO")
        .expect("set SCENARIO=<filename> to run this test");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/positions")
        .join(&file);
    let text = fs::read_to_string(&path).expect("readable epd");
    let case = parse_epd(&text);
    check_case(&case, &render::TerminalRenderer);
}