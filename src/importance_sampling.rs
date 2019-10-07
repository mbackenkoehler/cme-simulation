/// The dwSSA algorithm proposed by Daigle et al. [2011].
extern crate rand;

use self::rand::Rng;

use errors::*;
use model::*;
use progressbar::Progress;
use rare_event::{RareEvent, TimeInterval};

/// Estimate a rare event using the dwSSA algorithm. The rare event is given by
/// 'rare_event'. The probability is estimated using
/// importance sampling, where linear transformation parameters _gamma_ are
/// found using a cross-entropy method. At each of the cross-entropy iterations
/// 'K' simulations are run and 'rho' is the proportion, that determines the
/// _best_ trajectories.
pub fn estimate_rare_event_prob<RE: RareEvent>(
    model: &mut ReactionNetwork,
    rho: f64,
    k: usize,
    n: usize,
    rare_event: &RE,
) -> Result<f64> {
    set_gamma(model, rho, k, rare_event)?;
    info!("estimate rare event with {} simulations.", n);
    let mut sum = 0.0;
    let mut pb = Progress::new(n as u64)?;
    let s0: Vec<i32> = model
        .species
        .iter()
        .map(|s| *model.initial.get(&s.to_string()).unwrap_or(&0) as i32)
        .collect();
    for _ in 0..n {
        let mut s = s0.clone();
        let (distance, weight, _, _) = simulation_detailed(model, rare_event, false, &mut s)?;
        let event_weight = if distance == 0 { weight } else { 0.0 };
        pb.tick();
        sum += event_weight;
    }
    Ok(sum / n as f64)
}

/// An SSA version that uses a change-of-measure. This is assumed to be set via
/// the 'linear_bias' field in the 'Reaction' structs.
fn set_gamma<RE: RareEvent>(
    model: &mut ReactionNetwork,
    rho: f64,
    k: usize,
    rare_event: &RE,
) -> Result<()> {
    for i in 1.. {
        let best_size = (rho * k as f64).ceil() as usize;
        let mut best: Vec<(i32, f64, Vec<usize>, Vec<f64>)> = Vec::with_capacity(best_size + 1);

        // simulate k times and store the best trajectories
        let mut pb = Progress::new(k as u64)?;
        for _ in 1..k {
            let entry = simulation(model, rare_event, true)?;
            pb.tick();
            let pos = match best.binary_search_by_key(&entry.0, |&(d, _, _, _)| d) {
                Ok(pos) => pos,
                Err(pos) => pos,
            };
            if pos < best_size {
                best.insert(pos, entry);
                if best.len() > best_size {
                    best.pop();
                }
            }
        }
        drop(pb);

        // update biases
        let mut updated_bias = false;
        for (i, r) in model.reactions.iter_mut().enumerate() {
            let enumerator: f64 = best
                .iter()
                .map(|(_, weight, counts, _)| weight * counts[i] as f64)
                .sum();
            let denominator: f64 = best
                .iter()
                .map(|(_, weight, _, rweights)| weight * rweights[i])
                .sum();
            trace!("enum={:4.e} denum={:4.e}", enumerator, denominator);
            if denominator == 0.0 || enumerator == 0.0 {
                warn!("could not determine bias for reaction {}", i);
            } else {
                updated_bias = true;
                r.linear_bias = enumerator / denominator;
                if r.linear_bias.is_nan() {
                    bail!("illegal bias (nan) (= {} / {})", enumerator, denominator);
                }
            }
        }
        if !updated_bias {
            bail!(
                "no distance variation in samples; try increasing -k or change the target region"
            );
        }

        let total_distance: i32 = best.iter().map(|(distance, _, _, _)| distance).sum();

        info!(
            "CE search i={} max(distance(i))={} total_distance={}",
            i,
            best.last().unwrap().0,
            total_distance
        );
        debug!(
            "gamma={:?}",
            model
                .reactions
                .iter()
                .map(|r| r.linear_bias)
                .collect::<Vec<_>>()
        );

        if total_distance == 0 {
            return Ok(());
        }
    }
    unreachable!();
}

/// Runs a biased simulation and returns the change-of-measure weighting _w_ alongside
/// the distance to the rare event region _r_ as a tuple (_d_, _w_).
fn simulation<RE: RareEvent>(
    spec: &ReactionNetwork,
    rare_event: &RE,
    store_traj_info: bool,
) -> Result<(i32, f64, Vec<usize>, Vec<f64>)> {
    let mut s: Vec<i32> = spec
        .species
        .iter()
        .map(|s| *spec.initial.get(&s.to_string()).unwrap_or(&0) as i32)
        .collect();
    simulation_detailed(spec, rare_event, store_traj_info, &mut s)
}

fn simulation_detailed<RE: RareEvent>(
    spec: &ReactionNetwork,
    rare_event: &RE,
    store_traj_info: bool,
    s: &mut Vec<i32>,
) -> Result<(i32, f64, Vec<usize>, Vec<f64>)> {
    let spec = &*spec;
    let mut stack = Stack::default();
    let mut rng = self::rand::thread_rng();
    let mut t = 0f64;
    let stoich: Vec<Vec<i32>> = spec
        .reactions
        .iter()
        .map(|r| spec.species.iter().map(|s| r.change(s)).collect())
        .collect();
    let mut reaction_counter = vec![0; spec.reactions.len()];
    let mut rate_weights = vec![0.0; spec.reactions.len()];
    let mut weight = 1.0;
    let mut dt = 0.0;
    let mut distance = ::std::i32::MAX;
    let mut reached = false;
    loop {
        // compute rates
        let biased_rates: Vec<f64> = spec
            .reactions
            .iter()
            .map(|r| (*r).rate_program.eval(&s, &mut stack) * (*r).linear_bias)
            .collect();
        let biased_rate_sum: f64 = biased_rates.iter().sum();
        let unbiased_rate_sum: f64 = biased_rates
            .iter()
            .zip(spec.reactions.iter())
            .map(|(v, r)| v / r.linear_bias)
            .sum();

        // update tracking variables for CE
        if !reached && store_traj_info && t > 0.0 {
            for (i, rw) in rate_weights.iter_mut().enumerate() {
                *rw += biased_rates[i] / spec.reactions[i].linear_bias * dt;
            }
        }

        // set time change
        dt = -rng.gen::<f64>().ln() / biased_rate_sum;
        distance = distance.min(rare_event.distance(&TimeInterval::new_unchecked(t, t + dt), &s));
        reached |= distance == 0;

        let tnew = t + dt;

        // update time
        t = tnew;

        // select reaction
        let rsel: f64 = rng.gen::<f64>() * biased_rate_sum;
        let mut acc = 0f64;
        let mut sel = 0;
        for (i, rate_i) in biased_rates.iter().enumerate() {
            acc += *rate_i;
            if acc >= rsel {
                sel = i;
                break;
            }
        }

        // update weight
        if !reached {
            let bias_sel = spec.reactions[sel].linear_bias;
            weight *= (dt * (biased_rate_sum - unbiased_rate_sum)).exp() / bias_sel;
            if store_traj_info {
                reaction_counter[sel] += 1;
            }
        }

        // apply reaction or stop
        if t <= rare_event.max_time() {
            for (i, state_i) in s.iter_mut().enumerate() {
                *state_i += stoich[sel][i];
            }
        } else {
            break;
        }
    }
    Ok((distance, weight, reaction_counter, rate_weights))
}
