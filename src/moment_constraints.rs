use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use model::*;
use errors::*;
use model_parser;
use utils::Omega;

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct Moment {
    m: Vec<u32>,
}

impl Moment {
    pub fn from_vec(m: Vec<u32>) -> Self {
        Moment { m }
    }

    #[inline]
    pub fn monomial(&self, names: &[Ident], offsets: Option<&[f64]>) -> Result<Expr> {
        if names.len() != self.m.len() {
            bail!("illegal moment vector ({:?})", self.m);
        }
        use model::Expr::*;
        let mut monomial = Number(1.0);
        for (i, (exp, name)) in self.m.iter().zip(names).enumerate() {
            let mut var = Box::new(Id(name.to_string()));
            if let Some(delta) = offsets {
                let offset = Box::new(Number(delta[i]));
                var = Box::new(Op(var, Opcode::Add, offset));
            }
            let pow = Box::new(Number(f64::from(*exp)));
            let power = Box::new(Op(var, Opcode::Pow, pow));
            monomial = Op(Box::new(monomial), Opcode::Mul, power);
        }
        Ok(monomial)
    }

    #[inline]
    pub fn monomial_value(&self, state: &[i32]) -> f64 {
        state
            .iter()
            .zip(self.m.iter())
            .map(|(s, p)| s.pow(*p))
            .product::<i32>()
            .into()
    }

    pub fn as_vec(&self) -> Vec<u32> {
        self.m.clone()
    }
}

#[derive(PartialEq, Eq, Debug, Clone, Hash)]
pub struct TemporalMoment {
    moment: Moment,
    exponent: i32,
}

impl TemporalMoment {
    pub fn new(moment: Moment, exponent: i32) -> Self {
        TemporalMoment { moment, exponent }
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialMoment {
    moment: Moment,
    rho: f64,
}

impl ExponentialMoment {
    pub fn new(moment: Moment, rho: f64) -> Self {
        ExponentialMoment { moment, rho }
    }
}

#[derive(Debug, Clone)]
pub struct ExponentialInvMoment {
    moment: Moment,
    rho: f64,
    tmax: f64,
}

impl ExponentialInvMoment {
    pub fn new(moment: Moment, rho: f64, tmax: f64) -> Self {
        ExponentialInvMoment { moment, rho, tmax }
    }
}

#[derive(Debug, Clone)]
pub enum TransientMoment {
    Poly { tm: TemporalMoment },
    ExpInv { em: ExponentialInvMoment },
    Exp { em: ExponentialMoment },
}

impl TransientMoment {
    #[inline]
    fn time_integral_fact(&self, tlo: f64, thi: f64) -> f64 {
        use self::TransientMoment::*;
        match *self {
            Poly { ref tm } => {
                if tm.exponent == -1 {
                    1.0
                } else {
                    (thi.powi(tm.exponent + 1) - tlo.powi(tm.exponent + 1))
                        / f64::from(tm.exponent + 1)
                }
            }
            ExpInv { ref em } => {
                if em.rho == 0.0 {
                    thi - tlo
                } else {
                    (((em.rho * (em.tmax - tlo)).exp()) - ((em.rho * (em.tmax - thi)).exp()))
                        / em.rho
                }
            }
            Exp { ref em } => {
                if em.rho == 0.0 {
                    thi - tlo
                } else {
                    ((em.rho * thi).exp() - (em.rho * tlo).exp()) / em.rho
                }
            }
        }
    }

    #[inline]
    fn time_fact(&self, t: f64) -> f64 {
        use self::TransientMoment::*;
        match *self {
            Poly { ref tm } => t.powi(tm.exponent),
            ExpInv { ref em } => (em.rho * (em.tmax - t)).exp(),
            Exp { ref em } => (em.rho * t).exp(),
        }
    }

    #[inline]
    pub fn moment(&self) -> &Moment {
        use self::TransientMoment::*;
        match *self {
            Poly { ref tm } => &tm.moment,
            ExpInv { ref em } => &em.moment,
            Exp { ref em } => &em.moment,
        }
    }

    #[inline]
    pub fn parameter(&self) -> f64 {
        use self::TransientMoment::*;
        match *self {
            Poly { ref tm } => f64::from(tm.exponent),
            ExpInv { ref em } => em.rho,
            Exp { ref em } => -em.rho,
        }
    }

    #[inline]
    pub fn is_polynomial(&self) -> bool {
        use self::TransientMoment::*;
        match *self {
            Poly { .. } => true,
            ExpInv { .. } => false,
            Exp { .. } => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TemporalAccumulator {
    pub tm: TransientMoment,
    pub value: f64,
    pub usage: usize,
}

impl TemporalAccumulator {
    pub fn new(tm: TransientMoment) -> TemporalAccumulator {
        TemporalAccumulator {
            tm,
            value: 0.0,
            usage: 0,
        }
    }

    #[inline]
    pub fn count(&mut self, state: &[i32], tlo: f64, thi: f64) {
        let dt = self.tm.time_integral_fact(tlo, thi);
        let mval = self.tm.moment().monomial_value(state);
        self.value = mval.mul_add(dt, self.value);
    }

    pub fn clear(&mut self) {
        self.value = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct MomentConstraint {
    pub moment: TransientMoment,
    /// negative coefficients
    tm_coeffs: HashMap<usize, f64>,
    zero_term: f64,
}

impl MomentConstraint {
    fn new(
        model: &ReactionNetwork,
        moment: TransientMoment,
        pos: &mut AccuIds,
        tmax: f64,
        derivs: &mut HashMap<Moment, HashMap<Moment, f64>>,
    ) -> Result<MomentConstraint> {
        use self::TransientMoment::*;

        if !derivs.contains_key(moment.moment()) {
            derivs.insert(
                moment.moment().clone(),
                moment_deriv_coeffs(model, moment.moment())?,
            );
        }
        let coeffs = &derivs[moment.moment()];
        let mut tm_coeffs: HashMap<usize, f64> = HashMap::new();
        let mut zero_term = 0.0;
        match &moment {
            Poly { tm } => {
                for (m, v) in coeffs {
                    if m.as_vec().iter().all(|x| *x == 0) {
                        zero_term = v * moment.time_integral_fact(0.0, tmax);
                    } else {
                        let m_ = Poly {
                            tm: TemporalMoment::new(m.clone(), tm.exponent),
                        };
                        tm_coeffs.insert(pos.get_id(m_), *v);
                    }
                }
                let moment_ = Poly {
                    tm: TemporalMoment::new(tm.moment.clone(), tm.exponent - 1),
                };
                let p = pos.get_id(moment_);
                // This is due to the sympy simplification and the fact, that
                // the right-hand side integral has power 's - 1'.
                debug_assert!(!tm_coeffs.contains_key(&p));
                tm_coeffs.insert(p, tm.exponent.into());
            }
            ExpInv { em } => {
                for (m, v) in coeffs {
                    if m.as_vec().iter().all(|x| *x == 0) {
                        zero_term = v * moment.time_integral_fact(0.0, tmax);
                    } else {
                        let m_ = ExpInv {
                            em: ExponentialInvMoment::new(m.clone(), em.rho, em.tmax),
                        };
                        tm_coeffs.insert(pos.get_id(m_), *v);
                    }
                }
                let moment_ = ExpInv {
                    em: ExponentialInvMoment::new(em.moment.clone(), em.rho, em.tmax),
                };
                let p = pos.get_id(moment_);
                use std::collections::hash_map::Entry::*;
                match tm_coeffs.entry(p) {
                    Occupied(mut e) => {
                        *e.get_mut() -= em.rho;
                        pos.accus[p].usage -= 1;
                    }
                    Vacant(e) => {
                        e.insert(-em.rho);
                    }
                };
            }
            Exp { em } => {
                for (m, v) in coeffs {
                    if m.as_vec().iter().all(|x| *x == 0) {
                        zero_term = v * moment.time_integral_fact(0.0, tmax);
                    } else {
                        let m_ = Exp {
                            em: ExponentialMoment::new(m.clone(), em.rho),
                        };
                        tm_coeffs.insert(pos.get_id(m_), *v);
                    }
                }
                let moment_ = Exp {
                    em: ExponentialMoment::new(em.moment.clone(), em.rho),
                };
                let p = pos.get_id(moment_);
                use std::collections::hash_map::Entry::*;
                match tm_coeffs.entry(p) {
                    Occupied(mut e) => {
                        *e.get_mut() += em.rho;
                        pos.accus[p].usage -= 1;
                    }
                    Vacant(e) => {
                        e.insert(em.rho);
                    }
                };
            }
        }
        Ok(MomentConstraint {
            moment,
            tm_coeffs,
            zero_term,
        })
    }

    fn f(
        &self,
        t0: f64,
        t: f64,
        s0: &[i32],
        st: &[i32],
        accus: &[Option<TemporalAccumulator>],
    ) -> f64 {
        let mval_0: f64 = self.moment.time_fact(t0) * self.moment.moment().monomial_value(s0);
        let mval_t: f64 = self.moment.time_fact(t) * self.moment.moment().monomial_value(st);
        -self
            .tm_coeffs
            .iter()
            .map(|(tm, c)| c * accus[*tm].as_ref().expect("missing accumulator").value)
            .sum::<f64>()
            - mval_0
            + mval_t
            - self.zero_term
    }

    fn invalidate(&self, accus: &mut [Option<TemporalAccumulator>]) {
        for k in self.tm_coeffs.keys() {
            accus[*k].as_mut().unwrap().usage -= 1;
            if accus[*k].as_ref().unwrap().usage == 0 {
                accus[*k] = None;
            }
        }
    }
}

#[derive(Debug, Default)]
struct AccuIds {
    poly_map: HashMap<TemporalMoment, usize>,
    exp_map: HashMap<Moment, usize>,
    accus: Vec<TemporalAccumulator>,
}

impl AccuIds {
    fn get_id(&mut self, mom: TransientMoment) -> usize {
        use self::TransientMoment::*;
        let pos = match &mom {
            Poly { tm } => {
                if self.poly_map.contains_key(&tm) {
                    self.poly_map[&tm]
                } else {
                    self.accus.push(TemporalAccumulator::new(mom.clone()));
                    let p = self.accus.len() - 1;
                    self.poly_map.insert(tm.clone(), p);
                    p
                }
            }
            ec => {
                let m = ec.moment();
                if self.exp_map.contains_key(m) {
                    self.exp_map[m]
                } else {
                    self.accus.push(TemporalAccumulator::new(mom.clone()));
                    let p = self.accus.len() - 1;
                    self.exp_map.insert(m.clone(), p);
                    p
                }
            }
        };
        self.accus[pos].usage += 1;
        pos
    }

    fn reset_maps(&mut self) {
        self.exp_map.clear();
    }
}

#[derive(Clone)]
pub struct MomentConstraints {
    pub constraints: Vec<MomentConstraint>,
    accus: Vec<Option<TemporalAccumulator>>,
}

impl MomentConstraints {
    pub fn constraint_values<'a>(
        &'a self,
        t0: f64,
        t: f64,
        s0: &'a [i32],
        st: &'a [i32],
    ) -> impl Iterator<Item = f64> + 'a {
        self.constraints
            .iter()
            .map(move |c| c.f(t0, t, s0, st, &self.accus))
    }

    #[inline]
    pub fn count(&mut self, state: &[i32], tlo: f64, thi: f64) {
        self.accus.iter_mut().for_each(|a| {
            if let Some(v) = a.as_mut() {
                v.count(state, tlo, thi)
            }
        });
    }

    #[inline]
    pub fn clear_accus(&mut self) {
        self.accus.iter_mut().for_each(|a| {
            if let Some(v) = a.as_mut() {
                v.clear()
            }
        });
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn accu_len(&self) -> usize {
        self.accus.len()
    }

    pub fn remove(&mut self, idx: usize) {
        self.constraints[idx].invalidate(&mut self.accus);
        self.constraints.remove(idx);
    }

    pub fn defrag_accus(&mut self) {
        for (new_i, (ix, _u)) in self
            .accus
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if a.is_some() {
                    Some((i, a.as_ref().unwrap().usage))
                } else {
                    None
                }
            })
            .enumerate()
        {
            self.constraints.iter_mut().for_each(|c| {
                if let Some(coeff) = c.tm_coeffs.remove(&ix) {
                    c.tm_coeffs.insert(new_i, coeff);
                }
            });
        }
        self.accus.retain(Option::is_some);
    }
}

pub struct ConstraintBuilder {
    model: ReactionNetwork,
    constraints: Vec<MomentConstraint>,
    accus: AccuIds,
    moment_derivs: HashMap<Moment, HashMap<Moment, f64>>,
}

impl ConstraintBuilder {
    pub fn new(model: ReactionNetwork) -> Self {
        ConstraintBuilder {
            model,
            constraints: vec![],
            accus: AccuIds::default(),
            moment_derivs: HashMap::new(),
        }
    }

    pub fn with_polynomial_moments(
        &mut self,
        n: &[u32],
        s: &[i32],
        tmax: f64,
    ) -> Result<&mut Self> {
        let dim = self.model.species.len();
        for si in s {
            for ni in n {
                let mut omega2 = Omega::new(dim as u64, u64::from(*ni));
                while !omega2.done() {
                    let mut v = vec![0; dim];
                    omega2.set_val(&mut v);
                    let v = v.iter().map(|x| *x as u32).collect::<Vec<_>>();
                    let moment = Moment { m: v };
                    let tm = TemporalMoment {
                        moment,
                        exponent: *si,
                    };
                    let tm = TransientMoment::Poly { tm };
                    let mc = MomentConstraint::new(
                        &self.model,
                        tm,
                        &mut self.accus,
                        tmax,
                        &mut self.moment_derivs,
                    )?;
                    self.constraints.push(mc);
                    omega2.step();
                }
            }
        }
        Ok(self)
    }

    pub fn with_exponential_moments(
        &mut self,
        n: &[u32],
        rhos: &[f64],
        tmax: f64,
    ) -> Result<&mut Self> {
        let dim = self.model.species.len();
        for rho in rhos {
            for ni in n {
                let mut omega2 = Omega::new(dim as u64, u64::from(*ni));
                while !omega2.done() {
                    let mut v = vec![0; dim];
                    omega2.set_val(&mut v);
                    let v = v.iter().map(|x| *x as u32).collect::<Vec<_>>();
                    let moment = Moment { m: v };
                    let em = if *rho < 0.0 {
                        TransientMoment::ExpInv {
                            em: ExponentialInvMoment::new(moment, *rho, tmax),
                        }
                    } else {
                        TransientMoment::Exp {
                            em: ExponentialMoment::new(moment, -rho),
                        }
                    };
                    let mc = MomentConstraint::new(
                        &self.model,
                        em,
                        &mut self.accus,
                        tmax,
                        &mut self.moment_derivs,
                    )?;
                    self.constraints.push(mc);
                    omega2.step();
                }
            }
            self.accus.reset_maps();
        }
        Ok(self)
    }

    pub fn build(&mut self) -> MomentConstraints {
        MomentConstraints {
            constraints: self.constraints.drain(..).collect::<Vec<_>>(),
            accus: self.accus.accus.drain(..).map(Some).collect::<Vec<_>>(),
        }
    }
}

fn moment_deriv_coeffs(model: &ReactionNetwork, moment: &Moment) -> Result<HashMap<Moment, f64>> {
    let deriv = moment_deriv(model, &moment)
        .chain_err(|| format!("could not construct moment constraint for {:?}", &moment))?;
    let mut expanded_expr = expand_expr(&format!("{}", deriv))?;
    if expanded_expr.starts_with('-') {
        expanded_expr = format!("0 {}", expanded_expr);
    }
    let mut errors = Vec::new();
    match model_parser::ExprParser::new().parse(&mut errors, &expanded_expr) {
        Ok(expr) => {
            let mut coeffs = HashMap::new();
            extract_coeffs(&expr, model, &mut coeffs, false)?;
            Ok(coeffs)
        }
        Err(_) => bail!("could not parse '{}'", expanded_expr),
    }
}

fn extract_coeffs(
    expr: &Expr,
    model: &ReactionNetwork,
    coeffs: &mut HashMap<Moment, f64>,
    negative: bool,
) -> Result<()> {
    use model::Expr::*;
    match expr {
        Op(l, Opcode::Sub, r) => {
            extract_coeffs(&*l, model, coeffs, negative)?;
            extract_coeffs(&*r, model, coeffs, !negative)?;
        }
        Op(l, Opcode::Add, r) => {
            extract_coeffs(&*l, model, coeffs, negative)?;
            extract_coeffs(&*r, model, coeffs, negative)?;
        }
        e => {
            let (moment, coeff) = extract_coeff_mom(e, model)?;
            let counter = coeffs.entry(moment).or_insert(0.0);
            *counter += if negative { -coeff } else { coeff };
        }
    }
    Ok(())
}

fn extract_coeff_mom(expr: &Expr, model: &ReactionNetwork) -> Result<(Moment, f64)> {
    let mut pows = c! {n.to_string() => 0, for n in &model.species};
    let mut coeff = 1.0;
    extract_coeffs_term(expr, &mut pows, &mut coeff)?;
    let m = model.species.iter().map(|n| pows[n]).collect::<Vec<_>>();
    Ok((Moment { m }, coeff))
}

fn extract_coeffs_term(expr: &Expr, pows: &mut HashMap<Ident, u32>, coeff: &mut f64) -> Result<()> {
    use model::Expr::*;
    match expr {
        Op(l, Opcode::Mul, r) => {
            extract_coeffs_term(l, pows, coeff)?;
            extract_coeffs_term(r, pows, coeff)?;
        }
        Op(box Id(ref n), Opcode::Pow, box Number(e)) => {
            pows.insert(n.to_string(), *e as u32);
        }
        Number(f) => *coeff *= f,
        Id(ref n) => {
            pows.insert(n.to_string(), 1);
        }
        e => bail!("illegal subterm {:?}", e),
    }
    Ok(())
}

fn moment_deriv(model: &ReactionNetwork, moment: &Moment) -> Result<Expr> {
    use model::Expr::*;
    let mut deriv = Number(0.0);
    for reaction in &model.reactions {
        let prop = reaction.propensity.clone();
        let bias = Box::new(Number(reaction.linear_bias));
        let biased_prop = Box::new(Op(prop, Opcode::Mul, bias));
        let change = model
            .species
            .iter()
            .map(|n| f64::from(reaction.change(n)))
            .collect::<Vec<_>>();
        let shift_mom = Box::new(moment.monomial(&model.species, Some(&change))?);
        let mom = Box::new(moment.monomial(&model.species, None)?);
        let mom_factor = Box::new(Op(shift_mom, Opcode::Sub, mom));
        let term = Box::new(Op(mom_factor, Opcode::Mul, biased_prop));
        deriv = Op(Box::new(deriv), Opcode::Add, term);
    }
    Ok(deriv.subs(&model.constants))
}

fn expand_expr(expr: &str) -> Result<String> {
    // This causes some instability due to concurrency bugs with the GIL.
    let gil = Python::acquire_gil();
    let py = gil.python();
    let locals = PyDict::new(py);
    if let Ok(sympy) = py.import("sympy") {
        if locals.set_item("sympy", sympy).is_err() {
            bail!("python error setting locals");
        }
        let expr = format!("str(sympy.sympify('{}').expand())", expr);
        match py.eval(&expr, None, Some(&locals)) {
            Ok(res) => {
                if let Ok(res) = res.extract() {
                    Ok(res)
                } else {
                    bail!("error in expression simplification (extr)")
                }
            }
            Err(e) => {
                e.print_and_set_sys_last_vars(py);
                bail!("error in expression simplification")
            }
        }
    } else {
        bail!("could not import sympy")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! run_parser {
        ($parser_name:ident, $inp:expr) => {{
            let mut errors = Vec::new();
            model_parser::$parser_name::new()
                .parse(&mut errors, $inp)
                .expect("Parser failed unexpectedly")
        }};
    }

    macro_rules! birth_death {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    k_1 = 10
                    k_2 = 0.1
                species
                    X: int
                reactions
                    0 -> X @ k_1;
                    X -> 0 @ X * k_2;
                init
                    X = 0"
            )
        };
    }

    macro_rules! birth_dim {
        () => {
            run_parser!(
                ReactionNetworkParser,
                "
                parameters
                    k_1 = 10
                    k_2 = 0.1
                species
                    X: int
                reactions
                    0 -> X @ k_1;
                    2 X -> 0 @ (X - 1) * X * k_2;
                init
                    X = 0"
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

    macro_rules! do_until_ok {
        ($e:expr) => {
            loop {
                if let Ok(r) = $e {
                    break r;
                }
            }
        };
    }

    #[test]
    fn mom_extraction_bd_1() {
        let model = birth_death!();
        let mom1 = Moment { m: vec![1] };
        let mom0 = Moment { m: vec![0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom1));
        assert_eq!(res[&mom0], 10.0);
        assert_eq!(res[&mom1], -0.1);
    }

    #[test]
    fn mom_extraction_bd_2() {
        let model = birth_death!();
        let mom2 = Moment { m: vec![2] };
        let mom1 = Moment { m: vec![1] };
        let mom0 = Moment { m: vec![0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom2));
        assert_eq!(res[&mom0], 10.0);
        assert_eq!(res[&mom1], 20.1);
        assert_eq!(res[&mom2], -0.2);
    }

    #[test]
    fn mom_extraction_bdim_1() {
        let model = birth_dim!();
        let mom2 = Moment { m: vec![2] };
        let mom1 = Moment { m: vec![1] };
        let mom0 = Moment { m: vec![0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom1));
        assert_eq!(res[&mom0], 10.0);
        assert_eq!(res[&mom1], 0.2);
        assert_eq!(res[&mom2], -0.2);
    }

    #[test]
    fn mom_extraction_bdim_2() {
        let model = birth_dim!();
        let mom3 = Moment { m: vec![3] };
        let mom2 = Moment { m: vec![2] };
        let mom1 = Moment { m: vec![1] };
        let mom0 = Moment { m: vec![0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom2));
        assert_eq!(res[&mom0], 10.0);
        assert_eq!(res[&mom1], 19.6);
        assert_eq!(res[&mom2], 0.8);
        assert_eq!(res[&mom3], -0.4);
    }

    #[test]
    fn bdim_2d_1() {
        let model = birth_death_2d!();
        let mom00 = Moment { m: vec![0, 0] };
        let mom10 = Moment { m: vec![1, 0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom10));
        assert_eq!(res.len(), 2, "{:?}", res);
        assert_eq!(res[&mom00], 10.0);
        assert_eq!(res[&mom10], -0.1);
    }

    #[test]
    fn bdim_2d_2() {
        let model = birth_death_2d!();
        let mom00 = Moment { m: vec![0, 0] };
        let mom10 = Moment { m: vec![1, 0] };
        let mom20 = Moment { m: vec![2, 0] };
        let res = do_until_ok!(moment_deriv_coeffs(&model, &mom20));
        assert_eq!(res.len(), 3, "{:?}", res);
        assert_eq!(res[&mom00], 10.0);
        assert_eq!(res[&mom10], 20.1);
        assert_eq!(res[&mom20], -0.2);
    }
}
