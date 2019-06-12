#![allow(dead_code, unused_variables, unknown_lints)]
extern crate cme;
#[macro_use]
extern crate error_chain;
extern crate getopts;
extern crate lalrpop_util;
extern crate num_cpus;
#[macro_use]
extern crate log;
extern crate simplelog;

use getopts::{Matches, Options};
use simplelog::*;
use std::env;
use std::env::var;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::BufReader;
use std::str::FromStr;
use std::sync::Arc;

use cme::ast::ReactionNetwork;
use cme::errors::*;
use cme::importance_sampling::*;
use cme::model_parser::ReactionNetworkParser;
use cme::moment_estimation;
use cme::moment_estimation::estimate_expectations;
use cme::rare_event::*;
use cme::simulation;
use cme::simulation_logger::CsvSimulationLogger;
use cme::thread_pool::ThreadPool;

#[allow(clippy::needless_pass_by_value)]
fn str2level(inp: String) -> LevelFilter {
    match inp.to_uppercase().as_ref() {
        "TRACE" => LevelFilter::Trace,
        "DEBUG" => LevelFilter::Debug,
        "INFO" => LevelFilter::Info,
        "WARN" => LevelFilter::Warn,
        "ERROR" => LevelFilter::Error,
        _ => LevelFilter::Info,
    }
}

fn init_logger(log_file: &str) {
    let term_log_level = var("CME_TERM_LOG")
        .map(str2level)
        .unwrap_or(LevelFilter::Info);
    let file_log_level = var("CME_LOG").map(str2level).unwrap_or(LevelFilter::Debug);
    CombinedLogger::init(vec![
        TermLogger::new(term_log_level, Config::default())
            .expect("Initializing term logger failed"),
        WriteLogger::new(
            file_log_level,
            Config::default(),
            OpenOptions::new()
                .append(true)
                .create(true)
                .open(log_file)
                .expect("Initializing logger file failed"),
        ),
    ])
    .expect("Initializing logger failed");
}

fn print_usage(program: &str, opts: &Options) {
    let mut brief = format!(
        "Usage: {} [simulate|rare|means] MODEL [options...]\n\n",
        program
    );
    brief.push_str(
        "Modes:
    simulate  Stochastically simulate model.
    rare      Estimate a rare event probability.
    means     Estimate means",
    );

    print!("{}", opts.usage(&brief));
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Location {
    line: usize,
    column: usize,
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

fn from_pos(text: &str, pos: usize) -> Location {
    assert!(text.len() >= pos, "illegal parser error location");
    let mut cur_pos = 0;
    for (line_no, line) in text.lines().enumerate() {
        if cur_pos + line.len() >= pos {
            return Location {
                line: line_no + 1,
                column: pos - cur_pos + 1,
            };
        }
        cur_pos += line.len() + 1;
    }
    unreachable!("illegal parser error location");
}

fn read_model(file_name: &str) -> Result<ReactionNetwork> {
    let mut contents = String::new();
    let file = File::open(file_name).chain_err(|| format!("unable to open '{}'", file_name))?;
    let mut buf_reader = BufReader::new(file);
    buf_reader
        .read_to_string(&mut contents)
        .chain_err(|| "unable to read model file")?;
    use lalrpop_util::ParseError::*;
    let mut errors = Vec::new();
    let model: Result<ReactionNetwork> = match ReactionNetworkParser::new()
        .parse(&mut errors, &contents)
        .map_err(|e| e.map_location(|l| from_pos(&contents, l)))
    {
        Ok(v) => {
            v.name_check()?;
            Ok(v)
        }
        Err(InvalidToken { location }) => bail!("invalid token at location {}", location),
        Err(UnrecognizedToken { token, expected }) => {
            let exp = format!(
                "expected {}{}.",
                if expected.len() > 1 { "one of " } else { "" },
                expected.join(", ")
            );
            let (loc, tok, _) = token;
            bail!("unexpected token '{}' at {}; {}", tok, loc, exp);
        }
        _ => bail!("parser error"),
    };
    let mut model = model?; // The type checker needs this
    model.name = file_name.rsplit('/').next().map(String::from);
    Ok(model)
}

#[derive(Clone, Debug, PartialEq)]
enum Mode {
    Simulate,
    Rare,
    Means,
}

fn parse_args() -> Result<(ReactionNetwork, Mode, Matches)> {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();
    let mut opts = Options::new();
    opts.optopt("g", "granularity", "time granularity", "DT");
    opts.optopt("t", "tmax", "simulation duration", "T");
    opts.optopt("", "tmin", "minimum time (rare event)", "T");
    opts.optopt("d", "delta", "the state relevance threshold", "D");
    opts.optopt("o", "output", "set output file or directory", "NAME");
    opts.optopt("w", "warmup", "warmup before logging is started", "T");
    opts.optopt("r", "reruns", "number of reruns (>= 1)", "N");
    opts.optflagopt("j", "jobs", "number of jobs (>= 1) to use in parallel", "N");
    opts.optflag("", "notime", "do not write a time column");
    opts.optflag("", "write-id", "write a unique id for each rerun");
    opts.optmulti(
        "",
        "thres",
        "rare event threshold (always in connection with 'rare-var'). For a lower threshold give a positive integer. A negative value corresponds to an upper threshold. (rare event)",
        "VAL",
    );
    opts.optmulti(
        "",
        "rare-var",
        "rare event variable (always in connection with '[lower/upper]-thres'. (rare event)",
        "VAR",
    );
    opts.optopt(
        "",
        "rho",
        "the proportion of trajectories to use (rare event)",
        "VAl",
    );
    opts.optopt(
        "k",
        "k-rare",
        "the number of simulations for a cross-entropy iteration  (rare event)",
        "VAL",
    );
    opts.optopt(
        "n",
        "n-rare",
        "the number of simulations for the estimation (rare event)",
        "VAL",
    );
    opts.optopt(
        "e",
        "expectation",
        "variable for the estimation of its expected value",
        "VAR",
    );
    opts.optflag("", "lcv", "use linear control variates (LCV)");
    opts.optopt(
        "",
        "nlam",
        "number of lambdas sampled on [-5,5] (LCV)",
        "VAL",
    );
    opts.optopt(
        "",
        "pmin",
        "minimal correlation threshold max(max(p)/pmin,0.1)  (LCV)",
        "VAL",
    );
    opts.optopt(
        "",
        "prior",
        "lambda sampling distribution (LCV)",
        "{norm, unif}",
    );
    opts.optflag("", "prior-w0", "lambda set always includes 0 (LCV)");
    opts.optopt(
        "",
        "red-rule",
        "redundancy removal rule (LCV)",
        "{linear,cubic}",
    );
    opts.optopt("", "max-order", "maximum moment order (>0) (LCV)", "ORDER");
    opts.optflag("p", "plots", "draw plots of the results using pyplot");
    opts.optflag("h", "help", "print this help menu");
    if args.len() == 1 {
        print_usage(&program, &opts);
        bail!("no arguments provided");
    }
    let matches = opts
        .parse(&args[2..])
        .chain_err(|| "could not parse arguments")?;
    if matches.opt_present("h") {
        print_usage(&program, &opts);
        ::std::process::exit(0);
    }
    if args.len() < 3 {
        print_usage(&program, &opts);
        bail!("invalid options");
    }
    let mfile = &args[2];
    let mode = runtime_mode()?;
    let model = read_model(mfile)?;
    Ok((model, mode, matches))
}

fn runtime_mode() -> Result<Mode> {
    let args: Vec<String> = env::args().collect();
    match args[1].as_ref() {
        "simulate" => Ok(Mode::Simulate),
        "rare" => Ok(Mode::Rare),
        "means" => Ok(Mode::Means),
        _ => bail!("unknown mode '{}'", &args[1]),
    }
}

fn parse_num<T: FromStr>(matches: &Matches, opt: &str, default: T) -> Result<T> {
    match matches.opt_str(opt) {
        None => Ok(default),
        Some(str_val) => match str_val.parse::<T>() {
            Err(_) => bail!("-{} argument '{}' is not a valid number", opt, str_val),
            Ok(v) => Ok(v),
        },
    }
}

fn run_mode_simulation(model: ReactionNetwork, matches: &Matches) -> Result<()> {
    info!("Stochastic simulation mode");
    let granularity = parse_num(&matches, "g", 10f64)?;
    let tmax = parse_num(&matches, "t", 100f64)?;
    let warmup = parse_num(&matches, "w", 0f64)?;
    let r = parse_num(&matches, "r", 1usize)?;
    let write_time = !matches.opt_present("notime");
    let write_id = matches.opt_present("write-id");
    let o_file = matches.opt_str("o");
    let logger = CsvSimulationLogger::new(&model.species, o_file, write_time, write_id, r as u64)?;
    let settings = simulation::Settings::new(tmax, warmup, granularity, logger);
    let j = if matches.opt_present("j") {
        matches.opt_str("j").map_or_else(num_cpus::get, |s| {
            s.parse::<usize>().unwrap_or_else(|_| num_cpus::get())
        })
    } else {
        1
    };
    if j < 1 {
        bail!("illegal number of jobs ({})", j);
    }
    let mut pool = ThreadPool::new(j).chain_err(|| "could not initialize thread pool")?;
    let model = Arc::new(model);
    for id in 0..r {
        let settings_ = settings.clone();
        let model_ = Arc::clone(&model);
        pool.execute(move || simulation::simulation(id, &settings_, &model_))?;
    }
    pool.terminate()
        .chain_err(|| "could not terminate thread pool")?;
    drop(settings);
    if let Some(out_file) = matches.opt_str("o") {
        info!("-> Results have been written to {}.", out_file);
    }
    Ok(())
}

fn run_mode_rare(model: &mut ReactionNetwork, matches: &Matches) -> Result<()> {
    info!("Importance sampling (rare events) mode");
    let rare_event = read_rare_event(matches, model).chain_err(|| "parsing rare event failed")?;
    let rho = parse_num(matches, "rho", 0.01f64)?;
    let k = parse_num(matches, "k", 10_000usize)?;
    let n = parse_num(matches, "n", 100_000usize)?;
    let p_rare = estimate_rare_event_prob(model, rho, k, n, &rare_event)?;
    info!("Pr({}) ~ {:.4e}", rare_event, p_rare);
    Ok(())
}

fn read_rare_event(matches: &Matches, model: &ReactionNetwork) -> Result<CompositeEvent> {
    if !matches.opt_present("t") {
        bail!("a time horizon '-t' needs to be specified");
    }
    let tmax = parse_num(&matches, "t", 100f64)?;
    let time = match matches.opt_str("tmin") {
        Some(tmin) => {
            let tmin = tmin
                .parse::<f64>()
                .chain_err(|| format!("illegal tmin '{}'", tmin))?;
            EventTime::Interval(TimeInterval::new(tmin, tmax)?)
        }
        None => EventTime::Point(tmax),
    };
    let mut events = Vec::new();
    for (var, thres) in matches
        .opt_strs("rare-var")
        .iter()
        .zip(matches.opt_strs("thres"))
    {
        let var_idx = model
            .species
            .iter()
            .position(|v| var == v)
            .chain_err(|| format!("illegal variable '{}'", var))?;
        let thres = thres
            .parse::<i32>()
            .chain_err(|| format!("illegal threshold '{}'", thres))?;
        events.push(if thres >= 0 {
            ThresholdEvent::new_lower(time, var_idx, thres)
        } else {
            ThresholdEvent::new_upper(time, var_idx, thres.abs())
        })
    }
    if events.is_empty() {
        bail!("no rare event specified");
    }
    let event = CompositeEvent::new(events);
    if let Ok(e) = &event {
        debug!("read rare event {}", e);
    }
    event
}

fn run_mode_means(model: &mut ReactionNetwork, matches: &Matches) -> Result<()> {
    let tmax = parse_num(matches, "t", 100f64)?;
    let n = parse_num(matches, "n", 100_000usize)?;
    let lcv = matches.opt_present("lcv");
    let nlam = parse_num(matches, "nlam", 20)?;
    let pmin_frac = parse_num(matches, "pmin", 2.0)?;
    let reruns = parse_num(matches, "r", 1)?;
    let mut vars = matches.opt_strs("expectation");
    if vars.len() > 1 {
        bail!("more than one expectation not supported");
    }
    let prior: String = matches
        .opt_str("prior")
        .unwrap_or_else(|| "norm".to_string());
    let dist = match prior.as_str() {
        "norm" => moment_estimation::Distribution::Normal,
        "unif" => moment_estimation::Distribution::Uniform(-5.0, 5.0),
        d => bail!("unknown distribution: '{}', valid are 'unif' and 'norm'", d),
    };
    let prior = moment_estimation::LambdaPrior::new(dist, matches.opt_present("prior-w0"), nlam)?;
    let red_rule: String = matches
        .opt_str("red-rule")
        .unwrap_or_else(|| "linear".to_string());
    let red_rule = match red_rule.as_str() {
        "linear" => moment_estimation::RedundancyRule::UnscaledLinear,
        "cubic" => moment_estimation::RedundancyRule::ScaledCubic,
        "ucubic" => moment_estimation::RedundancyRule::UnscaledCubic,
        "thres" => moment_estimation::RedundancyRule::Threshold(0.99),
        r => bail!("invalid redundancy rule: '{}'", r),
    };
    let max_order = parse_num(matches, "max-order", 1u32)?;
    for v in &vars {
        if model.species.iter().all(|s| s != v) {
            bail!("invalid variable name: '{}'", v);
        }
    }
    if vars.is_empty() {
        vars = model.species.clone();
    }
    estimate_expectations(
        model,
        tmax,
        n,
        100,
        vars.clone(),
        !lcv,
        max_order,
        pmin_frac,
        &prior,
        &red_rule,
        reruns,
    )
}

fn run() -> Result<()> {
    let (mut model, mode, matches) = parse_args()?;
    match mode {
        Mode::Simulate => run_mode_simulation(model, &matches),
        Mode::Rare => run_mode_rare(&mut model, &matches),
        Mode::Means => run_mode_means(&mut model, &matches),
    }
}

fn main() {
    init_logger("run.log");
    if let Err(e) = run() {
        error!("error: {}", e);
        for e in e.iter().skip(1) {
            error!("caused by: {}", e);
        }

        // available when RUST_BACKTRACE=1
        if let Some(backtrace) = e.backtrace() {
            error!("backtrace: {:?}", backtrace);
        }
        ::std::process::exit(1);
    }
}
