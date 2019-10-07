use file_lock::FileLock;
use itertools::Itertools;
use nalgebra::base::{DMatrix, MatrixN};

use rand;
use rand::prelude::*;
use rand_distr::StandardNormal;

use model::*;
use covariance_accumulator::CovarianceAccumulator;
use errors::*;
use moment_constraints::*;
use progressbar::Progress;

#[derive(Debug, Clone)]
pub enum Distribution {
    Normal,
    Uniform(f64, f64),
}

impl Distribution {
    fn sample(&self) -> f64 {
        let mut rng = self::rand::thread_rng();
        use self::Distribution::*;
        match *self {
            Normal => rng.sample(StandardNormal),
            Uniform(a, b) => rng.gen::<f64>() * (b - a) + a,
        }
    }

    fn csv_id(&self) -> String {
        use self::Distribution::*;
        match *self {
            Normal => "norm".to_string(),
            Uniform(_, _) => "unif".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LambdaPrior {
    pub dist: Distribution,
    pub with_0: bool,
    nlam: usize,
}

impl LambdaPrior {
    pub fn new(dist: Distribution, with_0: bool, nlam: usize) -> Result<LambdaPrior> {
        if let self::Distribution::Uniform(a, b) = dist {
            if b <= a {
                bail!("Illegal uniform distribution on interval [{},{}]", a, b);
            }
        }
        if nlam < 1 {
            bail!("Illegal sample size: {}", nlam);
        }
        //let nlam = if with_0 { nlam - 1 } else { nlam };
        Ok(LambdaPrior { dist, with_0, nlam })
    }

    fn sample(&self) -> Vec<f64> {
        ::std::iter::once(if self.with_0 { 0.0 } else { self.dist.sample() })
            .chain((1..self.nlam).map(|_| self.dist.sample()))
            .collect::<Vec<_>>()
    }

    fn nlam(&self) -> usize {
        self.nlam
    }
}

pub fn estimate_expectations(
    model: &ReactionNetwork,
    time: f64,
    n: usize,
    d: usize,
    vars: Vec<String>,
    no_conds: bool,
    max_order: u32,
    pmin_frac: f64,
    prior: &LambdaPrior,
    red_rule: &RedundancyRule,
    reruns: usize,
) -> Result<()> {
    info!("{:?}", model.species);
    let mut rng = self::rand::thread_rng();
    let orders = if no_conds {
        vec![]
    } else {
        (1..=max_order).collect::<Vec<_>>()
    };
    debug!("orders={:?}", orders);
    let pos = model.species.iter().position(|s| s == &vars[0]).unwrap();
    let mut csb = ConstraintBuilder::new(model.clone());
    for _r in 0..reruns {
        let lambdas = prior.sample();
        debug!("lambdas={:?}", lambdas);
        csb.with_exponential_moments(&orders, &lambdas, time)?;
        let mut cs = csb.build();
        info!("{} LCV constraints", cs.len());
        debug!("{} accumulators", cs.accu_len());
        let mut cov = CovarianceAccumulator::new(cs.len() + 1);
        let s0 = model.initial_vector();
        let mut pb = Progress::new(n as u64)?;

        use std::time::Instant;
        let start_time = Instant::now();
        for iteration in 0..n {
            let mut s = s0.clone();
            simulation(&mut rng, model, time, &mut cs, &mut s)?;
            cov.step(
                cs.constraint_values(0.0, time, &s0, &s)
                    .chain(::std::iter::once(f64::from(s[pos]))),
            );
            cs.clear_accus();

            if iteration > 0 && iteration % d == 0 && !cs.is_empty() {
                cov = filter_constraints(&mut cs, cov, iteration, pmin_frac, red_rule);
            }

            pb.tick();
        }
        let duration = start_time.elapsed().as_nanos();

        ::std::mem::drop(pb);
        cov = filter_constraints(&mut cs, cov, n, pmin_frac, red_rule);
        debug!("constraint mean:\n{}", cov.mean);
        debug!("constraint variance:\n{}", cov.cov);
        let mut means = Vec::with_capacity(model.species.len());
        let mut means_lcv = Vec::with_capacity(model.species.len());
        let mut r2s = Vec::with_capacity(model.species.len());
        let corrs = cov.correlation();
        debug!("correlations: {}", &corrs);
        //let id = rng.gen::<u64>() as f64;
        let c_vec = cov.cov.column(cs.len());
        let c_vec = c_vec.slice((0, 0), (cs.len(), 1));
        /*
        use std::iter::{once, empty};
        for (j, c) in cs.constraints.iter().enumerate() {
            write_csv_line(
                "cvals.csv",
                empty()
                    .chain(c.moment.moment().as_vec().iter().map(|x| format!("{}", x)))
                    .chain(once(format!("{}", c.moment.parameter())))
                    .chain(once(format!("{}", corrs[(cs.len(), j)]))),
            )
            .chain_err(|| "error writing cvals.csv")?;
        }
        */
        let m = cov.cov.slice((0, 0), (cs.len(), cs.len()));
        let det = m.determinant().abs();
        debug!("det M = {:.4e}", det);
        if det < 1e-15 {
            warn!("Covariance matrix almost singular! det={:.4e}", det);
        }
        let idx = cs.len();
        let mean = cov.mean[idx];

        let m_inv = if det > 1e10 {
            debug!("scaling before inversion");
            let svec = m.diagonal().map(|x| 1.0 / x.sqrt());
            let s_inv = MatrixN::from_diagonal(&svec);
            let s_inv = s_inv.slice((0, 0), (cs.len(), cs.len()));
            let sm_inv = corrs
                .slice((0, 0), (cs.len(), cs.len()))
                .try_inverse()
                .chain_err(|| "inversion failed")?;
            s_inv * sm_inv * s_inv
        } else {
            m.try_inverse().chain_err(|| "inversion failed")?
        };
        debug!("inv M = {}", &m_inv);
        let beta = &m_inv * c_vec;
        let delta = beta.transpose() * cov.mean.slice((0, 0), (cs.len(), 1));
        means.push(mean);
        means_lcv.push(mean - delta[0]);
        let r2 = (&c_vec.transpose() * &m_inv * c_vec / cov.cov[(idx, idx)])[0];
        info!(
            "Variance reduction: {:.4e}",
            if !cs.is_empty() { r2 } else { 0.0 }
        );
        r2s.push(r2);
        debug!("means:{:?}", means);
        debug!("lcv_means:{:?}", means_lcv);

        // I won this round you stupid borrow checker....
        let model_name = model.name.as_ref().unwrap_or(&"none".to_string()).clone();
        let mut vals = vec![
            model_name,
            format!("{}", duration),
            format!("{}", means[0]),
            format!("{}", means_lcv[0]),
            format!("{}", no_conds),
            format!("{}", prior.nlam()),
            format!("{}", pmin_frac),
            format!("{:.4e}", r2s[0]),
            format!("{:.4e}", det),
            format!("{}", n),
            format!("{}", prior.dist.csv_id()),
            format!("{:?}", prior.with_0),
            format!("{:?}", red_rule),
            format!("{}", max_order),
            format!("{}", cs.len()),
            format!("{}", compute_pmin(&cov.correlation(), cs.len(), pmin_frac)),
        ];
        write_csv_line("summary.csv", vals.drain(..))
            .chain_err(|| "error writing 'summary.csv'")?;

        info!("{} CVs used.", cs.len());
        debug!("{:?}", cs.constraints);
        debug!(
            "lambdas={:?}",
            cs.constraints
                .iter()
                .map(|c| c.moment.parameter())
                .collect::<Vec<_>>()
        );

        info!("Means:");
        for (name, mean) in vars.iter().zip(means_lcv.iter()) {
            info!("  {}: {}", name, mean);
        }
    }

    Ok(())
}

#[derive(Debug)]
pub enum RedundancyRule {
    UnscaledLinear,
    ScaledCubic,
    UnscaledCubic,
    Threshold(f64),
}

impl RedundancyRule {
    #[inline]
    fn redundant(&self, p1: f64, p2: f64, pp: f64, p_min: f64) -> bool {
        use self::RedundancyRule::*;
        match self {
            UnscaledLinear => p1 + p2 < 2.0 * pp,
            ScaledCubic => 1.0 - ((1.0 - (p1 + p2) / 2.0) / (1.0 - p_min)).powi(2) < pp,
            UnscaledCubic => 1.0 - (1.0 - (p1 + p2) / 2.0).powi(2) < pp,
            Threshold(t) => t < &pp,
        }
    }
}

fn filter_constraints(
    cs: &mut MomentConstraints,
    mut cov: CovarianceAccumulator,
    _iteration: usize,
    pmin_frac: f64,
    red_rule: &RedundancyRule,
) -> CovarianceAccumulator {
    // iterate over constraints
    let corr = cov.correlation();
    let mut to_rm = vec![];
    let p_min = compute_pmin(&corr, cs.len(), pmin_frac);
    for c_idx in 0..cs.len() {
        let p = corr[(cov.dim() - 1, c_idx)].abs();
        //let p_ci = correlation_ci(p, iteration, 0.999);
        if cov.cov[(c_idx, c_idx)] < 1e-10 {
            to_rm.push(c_idx);
            continue;
        }
        if p < p_min {
            to_rm.push(c_idx);
            continue;
        }
        for c_idx_ in (c_idx + 1)..cs.len() {
            let p_ = corr[(cov.dim() - 1, c_idx_)].abs();
            let pp = corr[(c_idx, c_idx_)].abs();
            //if p + p_ < 2.0 * pp {
            if red_rule.redundant(p, p_, pp, p_min) {
                //debug!("c[{}] and c[{}] are redundant {} {} {}",
                //c_idx, c_idx_, p, p_, pp);
                if p_ > p {
                    to_rm.push(c_idx);
                    continue;
                } else {
                    to_rm.push(c_idx_);
                }
            }
        }
    }
    to_rm.sort();
    to_rm.dedup();
    for &idx in to_rm.iter().rev() {
        cs.remove(idx);
        cov = cov.remove(idx);
    }
    if !to_rm.is_empty() {
        //debug!(
        //"removing {} covariates in iteration {}",
        //to_rm.len(),
        //iteration
        //);
        cs.defrag_accus();
    }

    cov
}

#[inline]
fn compute_pmin(corr: &DMatrix<f64>, n_constraints: usize, pmin_frac: f64) -> f64 {
    (corr
        .column(n_constraints)
        .iter()
        .take(n_constraints)
        .filter(|x| !x.is_nan())
        .map(|x| x.abs())
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .unwrap_or(1.0)
        .min(1.0)
        / pmin_frac)
        .max(0.1)
}

fn write_csv_line(filename: &str, values: impl Iterator<Item = String>) -> ::std::io::Result<()> {
    use std::fs::OpenOptions;
    use std::io::prelude::*;
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .create(true)
        .open(filename)
        .unwrap();
    let lock = FileLock::lock(filename, true, true)?;
    for v in values.intersperse(String::from(",")) {
        write!(file, "{}", v)?;
    }
    writeln!(file)?;
    lock.unlock()
}

#[allow(dead_code)]
fn correlation_ci(r: f64, n: usize, p: f64) -> (f64, f64) {
    // Fisher transform and sd of 1 / sqrt(n - 1)
    let mu = r.atanh();
    let sd = 1.0 / (n as f64 - 3.0).sqrt();
    use statrs::function::erf::erf_inv;
    let q = ::std::f64::consts::SQRT_2 * erf_inv(2.0 * p - 1.0);
    ((mu - q * sd).tanh(), (mu + q * sd).tanh())
}

fn simulation<R: Rng>(
    rng: &mut R,
    spec: &ReactionNetwork,
    time: f64,
    constraints: &mut MomentConstraints,
    s: &mut Vec<i32>,
) -> Result<()> {
    let spec = &*spec;
    let mut stack = Stack::default();
    let mut t = 0f64;
    let stoich: Vec<Vec<i32>> = spec
        .reactions
        .iter()
        .map(|r| spec.species.iter().map(|s| r.change(s)).collect())
        .collect();
    loop {
        // compute rates
        let rates: Vec<f64> = spec
            .reactions
            .iter()
            .map(|r| (*r).rate_program.eval(&s, &mut stack))
            .collect();
        let rate_sum: f64 = rates.iter().sum();

        // set time change
        let dt = -rng.gen::<f64>().ln() / rate_sum;

        let tnew = t + dt;

        // update accumulators
        constraints.count(&s, t, tnew.min(time));

        // update time
        t = tnew;

        // select reaction
        let rsel: f64 = rng.gen::<f64>() * rate_sum;
        let mut acc = 0f64;
        let mut sel = 0;
        for (i, rate_i) in rates.iter().enumerate() {
            acc += *rate_i;
            if acc >= rsel {
                sel = i;
                break;
            }
        }

        // apply reaction or stop
        if t <= time {
            for (i, state_i) in s.iter_mut().enumerate() {
                *state_i += stoich[sel][i];
            }
        } else {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_parser;
    use test::Bencher;

    use self::rand::SeedableRng;

    macro_rules! run_parser {
        ($parser_name:ident, $inp:expr) => {{
            let mut errors = Vec::new();
            model_parser::$parser_name::new()
                .parse(&mut errors, $inp)
                .expect("Parser failed unexpectedly")
        }};
    }

    macro_rules! birth {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    a = 10

                vars
                    A : int

                reactions
                    0 -> A @ mass_action(a);

                init
                    A = 0"
            )
        };
    }

    macro_rules! birth_dim {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    a = 10
                    b = 0.1

                vars
                    A : int

                reactions
                    0 -> A @ mass_action(a);
                    2 A -> 0 @ mass_action(b);

                init
                    A = 0"
            )
        };
    }

    macro_rules! birth_death_2d {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    a = 10.0
                    b = 0.1
                species X Y
                reactions
                    0 -> X @ mass_action(a);
                    X -> 0 @ mass_action(b);
                    0 -> Y @ mass_action(a);
                    Y -> 0 @ mass_action(b);
                init
                    X = 0
                    Y = 0"
            )
        };
    }

    macro_rules! birth_dim_2d {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    a = 10.0
                    b = 0.1
                species X Y
                reactions
                    0   -> X @ mass_action(a);
                    2 X -> 0 @ mass_action(b);
                    0   -> Y @ mass_action(a);
                    2 Y -> 0 @ mass_action(b);
                init
                    X = 0
                    Y = 0"
            )
        };
    }

    macro_rules! dist_mod {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    k1 = 0.001
                    k2 = 0.001
                    k3 = 0.001
                    k4 = 0.001
                species X Y B
                reactions
                    X + Y -> B + Y @ mass_action(k1);
                    B + Y -> 2 Y   @ mass_action(k2);
                    Y + X -> B + X @ mass_action(k4);
                    B + X -> 2 X   @ mass_action(k3);
                init
                    X = 100
                    Y = 100
                    B = 100"
            )
        };
    }

    macro_rules! proc_mod {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    k = 0.01
                    SX = 0.01
                    SY = 0.01
                species X Y B C
                reactions
                    X + Y -> B + Y @ mass_action(k);
                    B + Y -> 2 Y   @ mass_action(k);
                    X -> B         @ mass_action(k * SY);
                    B -> Y         @ mass_action(k * SY);
                    Y + X -> C + X @ mass_action(k);
                    C + X -> 2 X   @ mass_action(k);
                    Y -> C         @ mass_action(k * SX);
                    C -> X         @ mass_action(k * SX);
                init
                    X = 10000
                    Y = 0
                    C = 0
                    B = 0"
            )
        };
    }

    macro_rules! bench_accus {
        ($model:expr, $m:expr, $s:expr, $me:expr, $rhos:expr, $test_name:ident) => {
            #[bench]
            #[ignore]
            fn $test_name(b: &mut Bencher) {
                let model = &$model;
                let time = 10.0;
                let mut csb = ConstraintBuilder::new(model.clone());
                csb.with_polynomial_moments($m, $s, time)
                    .unwrap()
                    .with_exponential_moments($me, $rhos, time)
                    .unwrap();
                let mut cs = csb.build();
                let mut rng = self::rand::rngs::StdRng::seed_from_u64(0);
                let mut cov = CovarianceAccumulator::new(cs.len() + model.species.len());
                let s0 = model.initial_vector();
                b.iter(|| {
                    let mut s = s0.clone();
                    simulation(&mut rng, model, time, &mut cs, &mut s).unwrap();
                    cov.step(
                        cs.constraint_values(0.0, time, &s0, &s)
                            .chain(s.iter().map(|x| f64::from(*x))),
                    );
                    cs.clear_accus();
                });
            }
        };
    }

    bench_accus!(birth!(), &[1], &[], &[], &[], bench_birth_s0);
    bench_accus!(birth!(), &[1], &[1], &[], &[], bench_birth_s1);
    bench_accus!(birth!(), &[1], &[1, 2], &[], &[], bench_birth_s2);
    bench_accus!(birth!(), &[1], &[1, 2, 3], &[], &[], bench_birth_s3);
    bench_accus!(birth!(), &[1], &[1, 2, 3, 4], &[], &[], bench_birth_s4);
    bench_accus!(birth!(), &[1], &[1, 2, 3, 4, 5], &[], &[], bench_birth_s5);

    bench_accus!(birth_dim!(), &[], &[], &[], &[], bench_birth_dim_0_accus);
    bench_accus!(birth_dim!(), &[1], &[1], &[], &[], bench_birth_dim_4_accus);
    bench_accus!(
        birth_dim!(),
        &[1],
        &[1, 2],
        &[],
        &[],
        bench_birth_dim_7_accus
    );
    bench_accus!(
        birth_dim!(),
        &[1],
        &[1, 2, 3],
        &[],
        &[],
        bench_birth_dim_10_accus
    );
    bench_accus!(
        birth_dim!(),
        &[1],
        &[1, 2, 3, 4],
        &[],
        &[],
        bench_birth_dim_ord1_s4
    );
    bench_accus!(
        birth_dim!(),
        &[1],
        &[1, 2, 3, 4, 5],
        &[],
        &[],
        bench_birth_dim_ord1_s5
    );
    bench_accus!(
        birth_dim!(),
        &[1, 2],
        &[1, 2],
        &[],
        &[],
        bench_birth_dim_10a_accus
    );
    bench_accus!(
        birth_dim!(),
        &[],
        &[1, 2],
        &[1, 2],
        &[0.0, -2.0],
        bench_birth_dim_2exp
    );

    bench_accus!(birth_death_2d!(), &[], &[], &[], &[], bench_2dbd_no_accus);
    bench_accus!(birth_death_2d!(), &[1], &[1], &[], &[], bench_2dbd_ord1_s1);
    bench_accus!(
        birth_death_2d!(),
        &[1],
        &[1, 2],
        &[],
        &[],
        bench_2dbd_ord1_s2
    );
    bench_accus!(
        birth_death_2d!(),
        &[1],
        &[1, 2, 3],
        &[],
        &[],
        bench_2dbd_ord1_s3
    );
    bench_accus!(
        birth_death_2d!(),
        &[1],
        &[1, 2, 3, 4],
        &[],
        &[],
        bench_2dbd_ord1_s4
    );

    bench_accus!(birth_dim_2d!(), &[], &[], &[], &[], bench_2dbdim_no_accus);
    bench_accus!(birth_dim_2d!(), &[1], &[1], &[], &[], bench_2dbdim_ord1_s1);
    bench_accus!(
        birth_dim_2d!(),
        &[1],
        &[1, 2],
        &[],
        &[],
        bench_2dbdim_ord1_s2
    );
    bench_accus!(
        birth_dim_2d!(),
        &[1],
        &[1, 2, 3],
        &[],
        &[],
        bench_2dbdim_ord1_s3
    );
    bench_accus!(
        birth_dim_2d!(),
        &[1],
        &[1, 2, 3, 4],
        &[],
        &[],
        bench_2dbdim_ord1_s4
    );

    bench_accus!(dist_mod!(), &[], &[], &[], &[], bench_dist_mod_no_accus);
    bench_accus!(dist_mod!(), &[1], &[1], &[], &[], bench_dist_mod_ord1_s1);
    bench_accus!(dist_mod!(), &[1, 2], &[1], &[], &[], bench_dist_mod_ord2_s1);
    bench_accus!(dist_mod!(), &[1], &[1, 2], &[], &[], bench_dist_mod_ord1_s2);
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3],
        &[],
        &[],
        bench_dist_mod_ord1_s3
    );
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3, 4],
        &[],
        &[],
        bench_dist_mod_ord1_s4
    );
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3, 4, 5],
        &[],
        &[],
        bench_dist_mod_ord1_s5
    );
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3, 4, 5, 6],
        &[],
        &[],
        bench_dist_mod_ord1_s6
    );
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3, 4, 5, 6, 7],
        &[],
        &[],
        bench_dist_mod_ord1_s7
    );
    bench_accus!(
        dist_mod!(),
        &[1, 2],
        &[1, 2, 3, 4],
        &[],
        &[],
        bench_dist_mod_ord2_s4
    );
    bench_accus!(
        dist_mod!(),
        &[1],
        &[1, 2, 3],
        &[1],
        &[0.0],
        bench_dist_mod_ord1_s3_exp1_rho1
    );

    macro_rules! bench_default_constraints {
        ($model:expr, $n: expr, $d: expr, $idcs: expr, $time:expr, $test_name:ident) => {
            bench_avg_detailed!(
                $model,
                $n,
                $d,
                $idcs,
                $time,
                build_default_constraints!(&$model, $time),
                $test_name
            );
        };
    }

    macro_rules! bench_empty_constraints {
        ($model:expr, $n: expr, $idcs: expr, $time:expr, $test_name:ident) => {
            bench_avg_detailed!(
                $model,
                $n,
                $n,
                $idcs,
                $time,
                build_empty_constraints!(&$model),
                $test_name
            );
        };
    }

    macro_rules! build_default_constraints {
        ($model: expr, $time:expr) => {{
            let mut csb = ConstraintBuilder::new($model.clone());
            csb.with_polynomial_moments(&[1, 2], &[1, 2, 3, 4, 5, 6], $time)
                .unwrap()
                .with_exponential_moments(
                    &[1, 2],
                    &[0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0],
                    $time,
                )
                .unwrap();
            csb.build()
        }};
    }

    macro_rules! build_empty_constraints {
        ($model: expr) => {
            ConstraintBuilder::new($model.clone()).build()
        };
    }

    macro_rules! bench_avg_detailed {
        ($model:expr, $n: expr, $d: expr, $idcs: expr, $time:expr, $constraints: expr, $test_name:ident) => {
            #[bench]
            #[ignore]
            fn $test_name(b: &mut Bencher) {
                let model = &$model;
                let mut rng = self::rand::rngs::StdRng::seed_from_u64(0);
                let s0 = model.initial_vector();
                let mut cs = $constraints;
                b.iter(|| {
                    let mut cov = CovarianceAccumulator::new(cs.len() + $idcs.len());
                    for iteration in 0..$n {
                        let mut s = s0.clone();
                        simulation(&mut rng, model, $time, &mut cs, &mut s).unwrap();
                        cov.step(
                            cs.constraint_values(0.0, $time, &s0, &s)
                                .chain($idcs.iter().map(|i: &usize| f64::from(s[*i]))),
                        );
                        cs.clear_accus();

                        if iteration > 0 && iteration % $d == 0 {
                            debug!("checking for redundant or weak constraints");
                            // iterate over constraints
                            let corr = cov.correlation();
                            let mut to_rm = vec![];
                            for c_idx in 0..cs.len() {
                                let p = corr[(cov.dim() - 1, c_idx)];
                                let p_ci = correlation_ci(p.abs(), iteration, 0.99);
                                for c_idx_ in (c_idx + 1)..cs.len() {
                                    let p_ci_ = correlation_ci(
                                        corr[(c_idx, c_idx_)].abs(),
                                        iteration,
                                        0.99,
                                    );
                                    if p_ci_.0 > 0.99 {
                                        debug!(
                                            "{} and {} highly correlated ({} v. {})",
                                            c_idx,
                                            c_idx_,
                                            p,
                                            corr[(cov.dim() - 1, c_idx_)]
                                        );
                                        if corr[(cov.dim() - 1, c_idx_)].abs() > p.abs() {
                                            to_rm.push(c_idx);
                                            continue;
                                        } else {
                                            to_rm.push(c_idx_);
                                        }
                                    }
                                }
                                if p_ci.1 < 0.2 {
                                    debug!("weak constraint: {} ({})", c_idx, p);
                                    to_rm.push(c_idx);
                                }
                            }
                            to_rm.sort();
                            to_rm.dedup();
                            for &idx in to_rm.iter().rev() {
                                cs.remove(idx);
                                cov = cov.remove(idx);
                            }
                        }
                    }
                });
            }
        };
    }

    bench_empty_constraints!(birth_dim!(), 1000, vec![0], 1.0, avg_birth_dim_1000);

    bench_default_constraints!(
        birth_dim!(),
        1000,
        100,
        vec![0],
        1.0,
        avg_lcv_birth_dim_1000_100
    );
    bench_empty_constraints!(dist_mod!(), 10000, vec![1], 50.0, avg_dist_mod_10000);

    bench_default_constraints!(
        dist_mod!(),
        10000,
        100,
        vec![1],
        50.0,
        avg_lcv_dist_mod_10000_100
    );

    bench_empty_constraints!(proc_mod!(), 10000, vec![2], 50.0, avg_proc_mod_10000);

    bench_default_constraints!(
        proc_mod!(),
        10000,
        100,
        vec![2],
        50.0,
        avg_lcv_proc_mod_10000_100
    );

    bench_empty_constraints!(proc_mod!(), 1000, vec![2], 50.0, avg_proc_mod_1000);

    bench_default_constraints!(
        proc_mod!(),
        1000,
        100,
        vec![2],
        50.0,
        avg_lcv_proc_mod_1000_100
    );

    bench_empty_constraints!(proc_mod!(), 500, vec![2], 50.0, avg_proc_mod_500);

    bench_default_constraints!(
        proc_mod!(),
        500,
        100,
        vec![2],
        50.0,
        avg_lcv_proc_mod_500_100
    );
}
