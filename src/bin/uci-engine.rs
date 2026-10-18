use chess_engine::engine::{AnyEval, EVALUATORS};
use chess_engine::uci::client::run_with_eval;

fn main() {
    // The evaluator can also be set at any time with `setoption name EvalFile`;
    // an argument only picks the one to start with.
    let eval = match std::env::args().nth(1) {
        None => AnyEval::default(),
        Some(name) => match AnyEval::parse_setting(&name) {
            Ok(eval) => eval,
            Err(e) => {
                eprintln!("uci-engine: {e}");
                eprintln!("usage: uci-engine [{}|<net file>]", EVALUATORS.join("|"));
                std::process::exit(2);
            }
        },
    };
    run_with_eval(eval);
}
