//! The abstract syntax tree of a CRN model.
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::hash::Hash;

use quickcheck::{Arbitrary, Gen};
use smallvec::SmallVec;

use errors::*;

pub const STACKSIZE: usize = 8;

pub type Ident = String;

pub enum SpeciesType {
    HighCount,
    LowCount,
}

/// A reaction network
#[derive(Clone)]
pub struct ReactionNetwork {
    /// Constant values
    pub constants: HashMap<Ident, f64>,

    /// Ordered species names
    pub species: Vec<Ident>,

    /// Reactions (order is constant)
    pub reactions: Vec<Reaction>,

    /// The (single) initial state
    pub initial: HashMap<Ident, usize>,

    /// An identifying name
    pub name: Option<String>,
}

macro_rules! sorted {
    ($v:ident) => {
        let mut $v = $v;
        $v.sort();
    };
}

impl ReactionNetwork {
    pub fn new(
        constants: HashMap<Ident, f64>,
        species: Vec<Ident>,
        mut reactions: Vec<Reaction>,
        initial: HashMap<Ident, usize>,
    ) -> ReactionNetwork {
        sorted!(species);
        let mut var_ids = HashMap::with_capacity(species.len());
        species.iter().enumerate().for_each(|(pos, name)| {
            var_ids.insert(name.to_string(), pos);
        });
        reactions
            .iter_mut()
            .for_each(|r| r.compile_rate(&var_ids, &constants));
        ReactionNetwork {
            species,
            constants,
            reactions,
            initial,
            name: None,
        }
    }

    pub fn name_check(&self) -> Result<()> {
        let mut names = self.species.clone();
        check_uniq(&names, "species")?;
        let constants: Vec<Ident> = self.constants.keys().cloned().collect();
        names.extend(constants);
        for (i, r) in self.reactions.iter().enumerate() {
            r.name_check(&names)
                .chain_err(|| format!("name error in reaction {}", i + 1))?;
        }
        for (i, id) in self.initial.keys().enumerate() {
            if !names.contains(id) {
                bail!("unknown identifier '{}' in initial value {}", id, i);
            }
        }
        Ok(())
    }

    pub fn initial_vector(&self) -> Vec<i32> {
        self.species
            .iter()
            .map(|s| self.initial[s] as i32)
            .collect()
    }

    pub fn dim(&self) -> usize {
        self.species.len()
    }

    pub fn stoichiometry(&self) -> Vec<Vec<i32>> {
        self.species
            .iter()
            .map(|s| {
                self.reactions
                    .iter()
                    .map(|r| r.change(s))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    pub fn stoichiometry_transposed(&self) -> Vec<Vec<i32>> {
        self.reactions
            .iter()
            .map(|r| self.species.iter().map(|s| r.change(s)).collect::<Vec<_>>())
            .collect()
    }
}

fn check_uniq<T: Eq + Clone + Hash + Display>(slice: &[T], kind: &str) -> Result<()> {
    let mut uniq = HashSet::new();
    for el in slice {
        if uniq.contains(el) {
            bail!("{} '{}' not unique", kind, el);
        }
        uniq.insert(el.clone());
    }
    Ok(())
}

/// A chemical reaction
#[derive(Clone)]
pub struct Reaction {
    /// Reaction input
    pub reactants: HashMap<Ident, usize>,

    /// Reaction output
    pub products: HashMap<Ident, usize>,

    /// Reaction propensity as a syntax tree
    pub propensity: Box<Expr>,

    /// An equivalent stack program for a more efficient evaluation of the
    /// propensity function.
    pub rate_program: StackProgram,

    /// A linear bias that is associated with this reaction.
    pub linear_bias: f64,
}

impl Reaction {
    fn name_check(&self, names: &[Ident]) -> Result<()> {
        self.propensity.name_check(names)?;
        for id in self.reactants.keys().chain(self.products.keys()) {
            if !names.contains(id) {
                bail!("unknown identifier: {}", id);
            }
        }
        Ok(())
    }

    fn compile_rate(&mut self, var_ids: &HashMap<Ident, usize>, subs: &HashMap<Ident, f64>) {
        self.rate_program = StackProgram::from_expr(&*self.propensity, var_ids, subs);
    }
}

#[allow(needless_pass_by_value)]
fn expand_mass_action(reactants: &HashMap<Ident, usize>, arg: Box<Expr>) -> Box<Expr> {
    let mut e = Expr::Number(1f64);
    for (species, amnt) in reactants {
        for i in 0..*amnt {
            let d = Expr::Op(
                Box::new(Expr::Id(species.to_string())),
                Opcode::Sub,
                Box::new(Expr::Number(i as f64)),
            );
            e = Expr::Op(Box::new(e), Opcode::Mul, Box::new(d));
        }
    }
    Box::new(Expr::Op(arg.clone(), Opcode::Mul, Box::new(e)))
}

impl Reaction {
    pub fn new(
        reactants: HashMap<Ident, usize>,
        products: HashMap<Ident, usize>,
        propensity: Box<Expr>,
    ) -> Reaction {
        let propensity = if let Expr::FuncApp(ref f_name, ref arg) = *propensity {
            if f_name == "mass_action" {
                expand_mass_action(&reactants, Box::new(*arg.clone()))
            } else {
                panic!(format!("Function \"{}\" not known.", f_name));
            }
        } else {
            propensity
        };
        Reaction {
            reactants,
            products,
            propensity,
            rate_program: StackProgram::empty(),
            linear_bias: 1.0,
        }
    }

    /// input of species `s` to the reaction
    pub fn input(&self, s: &str) -> usize {
        *self.reactants.get(s).unwrap_or(&0usize)
    }

    /// output of species `s` from to the reaction
    pub fn output(&self, s: &str) -> usize {
        *self.products.get(s).unwrap_or(&0usize)
    }

    /// net change of `s` due to the reaction
    pub fn change(&self, s: &str) -> i32 {
        let inp = self.input(s) as i32;
        let outp = self.output(s) as i32;
        outp - inp
    }
}

#[derive(Clone, Debug, Copy)]
pub enum Instruction {
    PushFloat(f64),
    PushVar(usize),
    Operation(Opcode),
}

#[derive(Default, Debug)]
pub struct Stack(SmallVec<[f64; STACKSIZE]>);

impl Stack {
    pub fn pop(&mut self) -> f64 {
        self.0.pop().unwrap()
    }

    pub fn push(&mut self, val: f64) {
        self.0.push(val);
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone)]
pub struct StackProgram(Vec<Instruction>);

impl StackProgram {
    fn empty() -> StackProgram {
        StackProgram(Vec::new())
    }

    fn from_expr(
        e: &Expr,
        var_ids: &HashMap<Ident, usize>,
        subs: &HashMap<Ident, f64>,
    ) -> StackProgram {
        let mut instructions = Vec::new();
        e.to_stack_program(&mut instructions, var_ids, subs);
        StackProgram(instructions)
    }

    pub fn eval_f64(&self, state: &[f64], stack: &mut Stack) -> f64 {
        assert!(stack.is_empty());
        let &StackProgram(ref instructions) = self;
        use self::Instruction::*;
        for instruction in instructions {
            match instruction {
                PushFloat(x) => stack.push(*x),
                PushVar(idx) => stack.push(state[*idx]),
                Operation(o) => {
                    let rv = stack.pop();
                    let lv = stack.pop();
                    stack.push(o.eval(lv, rv));
                }
            };
        }
        assert_eq!(stack.len(), 1);
        stack.pop()
    }

    pub fn eval(&self, state: &[i32], stack: &mut Stack) -> f64 {
        assert!(stack.is_empty());
        let &StackProgram(ref instructions) = self;
        use self::Instruction::*;
        for instruction in instructions {
            match instruction {
                PushFloat(x) => stack.push(*x),
                PushVar(idx) => stack.push(f64::from(state[*idx])),
                Operation(o) => {
                    let rv = stack.pop();
                    let lv = stack.pop();
                    stack.push(o.eval(lv, rv));
                }
            };
        }
        assert_eq!(stack.len(), 1);
        stack.pop()
    }
}

/// AST of an arithmetic expression of numbers and identifiers
#[derive(Clone)]
pub enum Expr {
    Number(f64),
    Id(String),
    FuncApp(String, Box<Expr>),
    Op(Box<Expr>, Opcode, Box<Expr>),
    Error,
}

impl Expr {
    pub fn evaluate(&self, ctx: &HashMap<Ident, f64>) -> Option<f64> {
        use self::Expr::*;
        match *self {
            Number(n) => Some(n),
            Op(ref l, op, ref r) => {
                if let (Some(lv), Some(rv)) = (l.evaluate(ctx), r.evaluate(ctx)) {
                    Some(op.eval(lv, rv))
                } else {
                    None
                }
            }
            Id(ref s) => ctx.get(s).cloned(),
            _ => None,
        }
    }

    pub fn subs(self, ctx: &HashMap<Ident, f64>) -> Expr {
        use self::Expr::*;
        match self {
            Op(l, op, r) => Op(Box::new(l.subs(ctx)), op, Box::new(r.subs(ctx))),
            FuncApp(f, a) => FuncApp(f.to_string(), Box::new(a.subs(ctx))),
            Id(ref s) if ctx.contains_key(s) => Number(ctx[s]),
            Id(s) => Id(s),
            Number(x) => Number(x),
            Error => Error,
        }
    }

    fn to_stack_program(
        &self,
        res: &mut Vec<Instruction>,
        var_ids: &HashMap<String, usize>,
        subs: &HashMap<Ident, f64>,
    ) {
        use self::Expr::*;
        use self::Instruction::*;
        match *self {
            Number(n) => res.push(PushFloat(n)),
            Op(ref l, op, ref r) => {
                l.to_stack_program(res, var_ids, subs);
                r.to_stack_program(res, var_ids, subs);
                res.push(Operation(op));
            }
            Id(ref s) => {
                if let Some(v) = subs.get(s) {
                    res.push(PushFloat(*v));
                } else {
                    let idx = *var_ids
                        .get(s)
                        .unwrap_or_else(|| panic!("Unexpected variable '{}'.", s));
                    res.push(PushVar(idx));
                }
            }
            _ => panic!("Compilation of '{:?}' failed.", self),
        };
    }

    fn name_check(&self, names: &[Ident]) -> Result<()> {
        use self::Expr::*;
        match *self {
            Number(_) => Ok(()),
            Op(ref l, _, ref r) => {
                l.name_check(names)?;
                r.name_check(names)
            }
            Id(ref s) => {
                if names.contains(s) {
                    Ok(())
                } else {
                    bail!("unknown identifier '{}' in propensity", s);
                }
            }
            _ => Ok(()),
        }
    }
}

/// Binary operations
#[derive(PartialEq, Copy, Clone)]
pub enum Opcode {
    Mul,
    Div,
    Add,
    Sub,
    Pow,
}

impl Arbitrary for Opcode {
    fn arbitrary<G: Gen>(g: &mut G) -> Self {
        use self::Opcode::*;
        let ops = [Mul, Div, Add, Sub];
        ops[(g.next_u32() % ops.len() as u32) as usize]
    }
}

impl Opcode {
    fn eval(self, lv: f64, rv: f64) -> f64 {
        use self::Opcode::*;
        match self {
            Mul => lv * rv,
            Div => lv / rv,
            Add => lv + rv,
            Sub => lv - rv,
            Pow => lv.powf(rv),
        }
    }
}

impl Display for Expr {
    fn fmt(&self, fmt: &mut Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        use self::Expr::*;
        match *self {
            Number(n) => {
                if (n.trunc() - n).abs() < 1e-6 {
                    write!(fmt, "{:?}", n.round() as i32)
                } else {
                    write!(fmt, "{:?}", n)
                }
            }
            Id(ref s) => write!(fmt, "{}", s),
            FuncApp(ref f, ref e) => write!(fmt, "{}({})", f, e),
            Op(ref l, op, ref r) => write!(fmt, "({} {:?} {})", l, op, r),
            Error => write!(fmt, "error"),
        }
    }
}

impl Debug for Expr {
    fn fmt(&self, fmt: &mut Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        use self::Expr::*;
        match *self {
            Number(n) => write!(fmt, "{:?}", n),
            Id(ref s) => write!(fmt, "{}", s),
            FuncApp(ref f, ref e) => write!(fmt, "{}({:?})", f, e),
            Op(ref l, op, ref r) => write!(fmt, "({:?} {:?} {:?})", l, op, r),
            Error => write!(fmt, "error"),
        }
    }
}

impl Debug for Opcode {
    fn fmt(&self, fmt: &mut Formatter) -> ::std::result::Result<(), ::std::fmt::Error> {
        use self::Opcode::*;
        match *self {
            Mul => write!(fmt, "*"),
            Div => write!(fmt, "/"),
            Add => write!(fmt, "+"),
            Sub => write!(fmt, "-"),
            Pow => write!(fmt, "**"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use test::Bencher;

    const TOL: f64 = 1e-15;

    macro_rules! compile {
        ($expr:ident, $vars:ident) => {{
            let ids = c! {v.to_string() => i, for (i, v) in $vars.iter().enumerate()};
            StackProgram::from_expr(&$expr, &ids, &HashMap::new())
        }};
    }

    macro_rules! check_equiv {
        ($expr:ident, $vars:ident, $vals:ident) => {{
            let ctx =
                c! {v.to_string() => f64::from(*x), for (v, x) in $vars.iter().zip($vals.iter())};
            let prog = compile!($expr, $vars);
            let mut stack = Stack::default();
            if let Some(v) = $expr.evaluate(&ctx) {
                (v - prog.eval($vals.as_slice(), &mut stack)).abs() < TOL
            } else {
                false
            }
        }};
    }

    macro_rules! bench_stackprogram {
        ($expression:expr, $bencher:ident) => {{
            let (expr, vars, vals) = $expression;
            let mut stack = Stack::default();
            let prog = compile!(expr, vars);
            $bencher.iter(|| prog.eval(&vals, &mut stack));
        }};
    }

    macro_rules! bench_expr_subs {
        ($expression:expr, $bencher:ident) => {{
            let (expr, vars, vals) = $expression;
            let ctx =
                c! {v.to_string() => f64::from(*x), for (v, x) in vars.iter().zip(vals.iter())};
            $bencher.iter(|| expr.evaluate(&ctx));
        }};
    }

    macro_rules! bench {
        ($expression:expr, $stack_bench:ident, $expr_bench:ident) => {
            #[bench]
            fn $stack_bench(b: &mut Bencher) {
                bench_stackprogram!($expression, b);
            }

            #[bench]
            fn $expr_bench(b: &mut Bencher) {
                bench_expr_subs!($expression, b);
            }
        };
    }

    fn expr_complex(opcode: Opcode, z: f64) -> (Expr, Vec<String>, Vec<i32>) {
        // <var x> <op> (<var y> - z)
        (
            Expr::Op(
                Box::new(Expr::Id("x".to_string())),
                opcode,
                Box::new(Expr::Op(
                    Box::new(Expr::Id("y".to_string())),
                    Opcode::Sub,
                    Box::new(Expr::Number(z)),
                )),
            ),
            c![v.to_string(), for v in vec!["x", "y"]],
            vec![47, 42],
        )
    }

    bench!(
        expr_complex(Opcode::Mul, 3.141),
        bench_complex_stack,
        bench_complex_expr
    );

    quickcheck! {
        fn prop_eval_complex(x: i32, opcode: Opcode, y: i32, z: f64) -> bool {
            if opcode == Opcode::Div && y == 1 {
                return true
            }
            let (expr, vars, _) = expr_complex(opcode, z);
            let vals = vec![x, y];
            check_equiv!(expr, vars, vals)
        }
    }

    fn expr_const_eval() -> (Expr, Vec<String>, Vec<i32>) {
        (
            Expr::Id("x".to_string()),
            c![v.to_string(), for v in vec!["x"]],
            vec![42],
        )
    }

    bench!(
        expr_const_eval(),
        bench_const_eval_stack,
        bench_const_eval_expr
    );

    quickcheck! {
        fn prop_const_eval_stack(x: i32) -> bool {
            let (expr, vars, _) = expr_const_eval();
            let vals = vec![x];
            check_equiv!(expr, vars, vals)
        }
    }

    quickcheck! {
        fn prop_const_eval(x: f64) -> bool {
            let (expr, _, _) = expr_const_eval();
            let mut ctx = HashMap::new();
            ctx.insert("x".to_string(), x);
            expr.evaluate(&ctx).map(|v| v == x).unwrap_or(false)
        }
    }

    fn expr_binop_1(opcode: Opcode, y: f64) -> (Expr, Vec<String>, Vec<i32>) {
        // <var x> <op> <float>
        (
            Expr::Op(
                Box::new(Expr::Id("x".to_string())),
                opcode,
                Box::new(Expr::Number(y)),
            ),
            vec!["x".to_string()],
            vec![53],
        )
    }

    bench!(
        expr_binop_1(Opcode::Div, 3.141),
        bench_binop_1_stack,
        bench_binop_1_expr
    );

    quickcheck! {
        fn prop_eval_binop_1(x: i32, opcode: Opcode, y: f64) -> bool {
            let (expr, vars, _) = expr_binop_1(opcode, y);
            let vals = vec![x];
            check_equiv!(expr, vars, vals)
        }
    }

    fn expr_binop_2(opcode: Opcode) -> (Expr, Vec<String>, Vec<i32>) {
        // <var x> <op> <var y>
        (
            Expr::Op(
                Box::new(Expr::Id("x".to_string())),
                opcode,
                Box::new(Expr::Id("y".to_string())),
            ),
            c![v.to_string(), for v in vec!["x", "y"]],
            vec![47, 11],
        )
    }

    bench!(
        expr_binop_2(Opcode::Div),
        bench_binop_2_stack,
        bench_binop_2_expr
    );

    quickcheck! {
        fn prop_eval_binop_2(x: i32, opcode: Opcode, y: i32) -> bool {
            if opcode == Opcode::Div && y == 0 {
                return true
            }
            let (expr, vars, _) = expr_binop_2(opcode);
            let vals = vec![x, y];
            check_equiv!(expr, vars, vals)
        }
    }
}
