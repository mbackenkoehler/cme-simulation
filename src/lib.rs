#![allow(unknown_lints, clippy::useless_attribute, renamed_and_removed_lints)]
#![feature(test, specialization, box_patterns)]
extern crate test;
#[macro_use]
extern crate error_chain;
extern crate rgsl;
extern crate smallvec;
extern crate statrs;
#[allow(unused_imports)]
#[macro_use]
extern crate quickcheck;
#[allow(unused_imports)]
#[macro_use(c)]
extern crate cute;
extern crate nalgebra;
#[macro_use]
extern crate log;
extern crate file_lock;
extern crate itertools;
extern crate lalrpop_util;
extern crate pyo3;
extern crate rand;

pub mod ast;
pub mod importance_sampling;
pub mod model_parser;
pub mod simulation;
pub mod simulation_logger;
pub mod thread_pool;
pub mod errors {
    error_chain! {}
}
pub mod covariance_accumulator;
pub mod distribution;
pub mod moment_constraints;
pub mod moment_estimation;
pub mod progressbar;
pub mod rare_event;
pub mod utils;

#[macro_export]
macro_rules! run_parser {
    ($parser_name:ident, $inp:expr) => {{
        let mut errors = Vec::new();
        model_parser::$parser_name::new()
            .parse(&mut errors, $inp)
            .expect("Parser failed unexpectedly")
    }};
}

#[macro_export]
macro_rules! parse_model {
    ($inp:expr) => {
        run_parser!(ReactionNetworkParser, $inp)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_parser {
        ($parser_name:ident, $inp:expr, $exp:expr) => {
            let parsed = run_parser!($parser_name, $inp);
            assert_eq!(&format!("{:?}", parsed).replace(".0", ""), $exp);
        };
    }

    macro_rules! test_expr_parser {
        ($inp:expr, $exp:expr) => {
            test_parser!(ExprParser, $inp, $exp);
        };
    }

    #[test]
    fn parse_whole_model_0() {
        let spec = "
            parameters
                k_1 = 10
                k_2 = 0.1
            species
                X: int
            reactions
                0 -> X @ k_1;
                X -> 0 @ X * k_2;
            init
                X = 0";
        run_parser!(ReactionNetworkParser, spec);
    }

    #[test]
    fn parse_reaction_0() {
        let res = run_parser!(ReactionParser, "0 -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 0usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_1() {
        let res = run_parser!(ReactionParser, "X -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 1usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_2() {
        let res = run_parser!(ReactionParser, "2 X -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 2usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_3() {
        let res = run_parser!(ReactionParser, "X + X -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 2usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_4() {
        let res = run_parser!(ReactionParser, "2 X + X + Y -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 3usize);
        assert_eq!(res[0].input("Y"), 1usize);
        assert_eq!(res[0].output("Y"), 0usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_5() {
        let res = run_parser!(ReactionParser, "X -> X @ 0;");
        assert_eq!(res[0].input("X"), 1usize);
        assert_eq!(res[0].output("X"), 1usize);
    }

    #[test]
    fn parse_reaction_6() {
        let res = run_parser!(ReactionParser, "X, X -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 2usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_7() {
        let res = run_parser!(ReactionParser, "2 X, Y + X + Y -> 0 @ 0;");
        assert_eq!(res[0].input("X"), 3usize);
        assert_eq!(res[0].input("Y"), 2usize);
        assert_eq!(res[0].output("Y"), 0usize);
        assert_eq!(res[0].output("X"), 0usize);
    }

    #[test]
    fn parse_reaction_8() {
        let res = run_parser!(ReactionParser, "2 X, Y + X + Y <- 0 @ 0;");
        assert_eq!(res[0].output("X"), 3usize);
        assert_eq!(res[0].output("Y"), 2usize);
        assert_eq!(res[0].input("Y"), 0usize);
        assert_eq!(res[0].input("X"), 0usize);
    }

    #[test]
    fn parse_species_section_0() {
        let res = run_parser!(SpeciesSectionParser, "species");
        assert_eq!(res.len(), 0);
    }

    #[test]
    fn parse_species_section_1() {
        let res = run_parser!(SpeciesSectionParser, "species a b c");
        assert_eq!(res.len(), 3);
        assert!(vec!["a", "b", "c"]
            .into_iter()
            .all(|s| res.contains(&s.to_string())));
    }

    #[test]
    fn parse_species_section_2() {
        let res = run_parser!(SpeciesSectionParser, "species a : int b: bool c");
        assert_eq!(res.len(), 3);
        assert!(vec!["a", "b", "c"]
            .into_iter()
            .all(|s| res.contains(&s.to_string())));
    }

    #[test]
    fn parse_parameter_declaration_section_0() {
        let res = run_parser!(ParameterSectionParser, "parameters");
        assert_eq!(res.len(), 0);
    }

    #[test]
    fn parse_parameter_declaration_section_1() {
        let res = run_parser!(ParameterSectionParser, "parameters a = 1");
        assert_eq!(res.len(), 1);
        assert_eq!(res["a"], 1f64);
    }

    #[test]
    fn parse_parameter_declaration_section_2() {
        let res = run_parser!(ParameterSectionParser, "parameters a = 1 b= 0.03");
        assert_eq!(res.len(), 2);
        assert_eq!(res["a"], 1f64);
        assert_eq!(res["b"], 0.03f64);
    }

    #[test]
    fn parse_basic() {
        test_expr_parser!("22", "22");
        test_expr_parser!("22 + 3", "(22 + 3)");
        test_expr_parser!("22 * 3", "(22 * 3)");
        test_expr_parser!("22 / 3", "(22 / 3)");
    }

    #[test]
    fn parse_basic_floats() {
        test_expr_parser!("22.5", "22.5");
        test_expr_parser!("22.0", "22");
    }

    #[test]
    fn parse_ident() {
        test_expr_parser!("b", "b");
        test_expr_parser!("asd", "asd");
        test_expr_parser!("asd2", "asd2");
        test_expr_parser!("as_d2", "as_d2");
        test_expr_parser!("_", "_");
        test_expr_parser!("_1", "_1");
    }

    #[test]
    fn parse_power() {
        test_expr_parser!("22 ^ 3", "(22 ** 3)");
        test_expr_parser!("22 ** 3", "(22 ** 3)");
        test_expr_parser!("2 * 22 ** 3", "(2 * (22 ** 3))");
    }

    #[test]
    fn parse_op_precedence() {
        test_expr_parser!("0 * (22 + 3)", "(0 * (22 + 3))");
        test_expr_parser!("0 * 22 + 3", "((0 * 22) + 3)");
        test_expr_parser!("1 ** 2 * 22 + 3", "(((1 ** 2) * 22) + 3)");
    }

    #[test]
    fn parse_fn_app() {
        test_expr_parser!("f(0)", "f(0)");
        test_expr_parser!("f(x)", "f(x)");
        test_expr_parser!("f(x * 2)", "f((x * 2))");
    }
}
