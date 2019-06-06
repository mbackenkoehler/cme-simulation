use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::Sized;
use std::rc::Rc;

type State = Vec<i32>;
type StateRef = Rc<State>;

#[derive(Default, Debug)]
pub struct Distribution<V: Value + Debug> {
    values: HashMap<StateRef, Rc<V>>,
}

pub trait Value
where
    Self: Sized,
{
    fn prob(&self) -> f64;

    fn pred(&self) -> Vec<Neighbor<Self>>;

    fn succ(&self) -> Vec<Neighbor<Self>>;

    fn exit_rate(&self) -> f64 {
        self.succ().iter().map(|n| n.rate).sum::<f64>()
    }
}

#[allow(clippy::ptr_arg)]
impl<V: Value + Debug> Distribution<V> {
    pub fn from_map(values: HashMap<StateRef, Rc<V>>) -> Distribution<V> {
        Distribution { values }
    }

    pub fn get(&self, state: &State) -> Option<&Rc<V>> {
        self.values.get(state)
    }

    pub fn get_mut(&mut self, state: &State) -> Option<&mut Rc<V>> {
        self.values.get_mut(state)
    }

    pub fn insert(&mut self, state: StateRef, value: V) {
        self.values.insert(state, Rc::new(value));
    }

    pub fn remove(&mut self, state: &State) {
        self.values.remove(state);
    }

    pub fn prob(&mut self, cond: impl Fn(&State) -> bool) -> f64 {
        self.values
            .iter()
            .map(|(s, v)| if cond(s) { v.prob() } else { 0.0 })
            .sum::<f64>()
    }

    pub fn values(&self) -> impl Iterator<Item = &Rc<V>> {
        self.values.values()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&StateRef, &Rc<V>)> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&StateRef, &mut Rc<V>)> {
        self.values.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug)]
pub struct Neighbor<V: Value> {
    pub value: Option<Rc<V>>,
    pub rate: f64,
    pub reaction_id: usize,
    pub ty: NeighborTy,
}

impl<V: Value> Neighbor<V> {
    pub fn new_pre(value: Option<&Rc<V>>, rate: f64, reaction_id: usize) -> Neighbor<V> {
        Neighbor {
            value: value.map(|v| Rc::clone(v)),
            rate,
            reaction_id,
            ty: NeighborTy::Predecessor,
        }
    }

    pub fn new_suc(value: Option<&Rc<V>>, rate: f64, reaction_id: usize) -> Neighbor<V> {
        Neighbor {
            value: value.map(|v| Rc::clone(v)),
            rate,
            reaction_id,
            ty: NeighborTy::Successor,
        }
    }
}

impl<V: Value> Clone for Neighbor<V> {
    fn clone(&self) -> Neighbor<V> {
        Neighbor {
            value: self.value.as_ref().map(Rc::clone),
            rate: self.rate,
            reaction_id: self.reaction_id,
            ty: self.ty,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NeighborTy {
    Successor,
    Predecessor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(PartialEq, Debug, Default)]
    struct V(f64);

    impl Value for V {
        fn prob(&self) -> f64 {
            self.0
        }
        fn pred(&self) -> Vec<Neighbor<V>> {
            unimplemented!();
        }
        fn succ(&self) -> Vec<Neighbor<V>> {
            unimplemented!();
        }
    }

    #[test]
    fn add_contains() {
        let mut dist = Distribution::default();
        let state = vec![];
        dist.insert(Rc::new(state), V(0.0));
        let res = dist.get(&vec![]);
        assert!(res.is_some());
        assert_eq!(*res.unwrap(), Rc::new(V(0.0)));
    }

    #[test]
    fn add_remove_1() {
        let mut dist = Distribution::default();
        let state = vec![];
        dist.insert(Rc::new(state), V(0.0));
        dist.remove(&vec![]);
        assert!(dist.get(&vec![]).is_none());
    }

    #[test]
    fn add_remove_2() {
        let mut dist = Distribution::default();
        let state = vec![1];
        dist.insert(Rc::new(state), V(0.0));
        dist.remove(&vec![]);
        let res = dist.get(&vec![1]);
        assert!(res.is_some());
        assert_eq!(*res.unwrap(), Rc::new(V(0.0)));
    }

    #[test]
    fn cond_prob_1() {
        let mut dist = Distribution::default();
        let s1 = Rc::new(vec![1]);
        let v1 = V(0.5);
        let cond = |s: &State| s == &vec![1];
        assert!(cond(&*s1));
        dist.insert(s1, v1);
        assert_eq!(dist.prob(cond), 0.5);
    }

    #[test]
    fn cond_prob_2() {
        let mut dist = Distribution::default();
        let s1 = Rc::new(vec![2]);
        let v1 = V(0.5);
        let cond = |s: &State| s == &vec![1];
        dist.insert(s1, v1);
        assert_eq!(dist.prob(cond), 0.0);
    }

    #[test]
    fn cond_prob_3() {
        let mut dist = Distribution::default();
        let s1 = Rc::new(vec![1]);
        let v1 = V(0.5);
        let cond = |s: &State| s == &vec![1];
        assert!(cond(&*s1));
        dist.insert(s1, v1);
        dist.insert(Rc::new(vec![2]), V(0.5));
        assert_eq!(dist.prob(cond), 0.5);
    }
}
