extern crate rand;

use errors::*;
use model::{ReactionNetwork, Stack};

#[derive(Clone)]
pub struct Settings<L: SimulationLogger> {
    pub tmax: f64,
    pub warmup: f64,
    pub granularity: f64,
    pub logger: L,
}

impl<L: SimulationLogger> Settings<L> {
    pub fn new(tmax: f64, warmup: f64, granularity: f64, logger: L) -> Settings<L> {
        Settings {
            tmax,
            warmup,
            granularity,
            logger,
        }
    }
}

pub trait SimulationLogger {
    fn log(&self, id: usize, time: f64, state: &[i32]) -> Result<()>;
    fn job_done(&self, id: usize, state: &[i32]) -> Result<()>;
}

pub fn simulation<L: SimulationLogger>(
    id: usize,
    settings: &Settings<L>,
    spec: &ReactionNetwork,
) -> Result<()> {
    use simulation::rand::Rng;
    let mut stack = Stack::default();
    //use self::rand::SeedableRng;
    //let mut rng = self::rand::rngs::StdRng::seed_from_u64(0);// self::rand::thread_rng();
    let mut rng = self::rand::thread_rng();
    let mut t = 0f64;
    let mut tlast = 0f64;
    let mut s: Vec<i32> = spec
        .species
        .iter()
        .map(|s| *spec.initial.get(&s.to_string()).unwrap_or(&0) as i32)
        .collect();
    let stoich: Vec<Vec<i32>> = spec
        .reactions
        .iter()
        .map(|r| spec.species.iter().map(|s| r.change(s)).collect())
        .collect();
    while t < settings.tmax {
        let rates: Vec<f64> = spec
            .reactions
            .iter()
            .map(|r| (*r).rate_program.eval(&s, &mut stack))
            .collect();
        let rate_sum: f64 = rates.iter().sum();
        t -= rng.gen::<f64>().ln() / rate_sum;
        while tlast - t.min(settings.tmax) < 1e-6 {
            if tlast < settings.warmup {
                continue;
            }
            settings.logger.log(id, tlast, &s)?;
            tlast += settings.granularity;
        }
        let rsel: f64 = rng.gen::<f64>() * rate_sum;
        let mut acc = 0f64;
        let mut sel = 0usize;
        for (i, rate_i) in rates.iter().enumerate() {
            acc += *rate_i;
            if acc >= rsel {
                sel = i;
                break;
            }
        }
        for (i, state_i) in s.iter_mut().enumerate() {
            *state_i += stoich[sel][i];
        }
    }
    settings.logger.job_done(id, &s)
}
