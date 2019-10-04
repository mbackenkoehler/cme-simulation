use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct CovarianceAccumulator {
    pub mean: DVector<f64>,
    pub cov: DMatrix<f64>,
    pub n: u32,
}

impl CovarianceAccumulator {
    pub fn new(dim: usize) -> Self {
        CovarianceAccumulator {
            mean: DVector::zeros(dim),
            cov: DMatrix::zeros(dim, dim),
            n: 0,
        }
    }

    pub fn step(&mut self, cvals: impl Iterator<Item = f64>) {
        self.n += 1;
        let v = DVector::from_iterator(self.mean.len(), cvals);
        let x = &v - &self.mean;
        let nm1 = f64::from(self.n - 1);
        let nf = f64::from(self.n);
        let alpha = nm1 / nf;
        self.cov.ger(alpha, &x, &x, nm1);
        self.cov /= nf;
        self.mean.axpy(1.0 / nf, &v, alpha);
    }

    pub fn correlation(&self) -> DMatrix<f64> {
        let v = self.cov.diagonal().map(f64::sqrt);
        let sdev = &v * &v.transpose();
        self.cov.component_div(&sdev)
    }

    pub fn remove(self, index: usize) -> Self {
        debug_assert!(
            index < self.mean.len(),
            "illegal removal index '{}' with dim={}",
            index,
            self.dim()
        );
        let cov = self.cov.remove_column(index);
        let cov = cov.remove_row(index);
        let mean = self.mean.remove_row(index);
        CovarianceAccumulator {
            cov,
            mean,
            n: self.n,
        }
    }

    pub fn dim(&self) -> usize {
        debug_assert!(self.mean.len() == self.cov.ncols());
        debug_assert!(self.cov.is_square());
        self.mean.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate rand;
    use self::rand::distributions::StandardNormal;
    use self::rand::prelude::*;

    #[test]
    fn std_normal() {
        let n = 100_000;
        let v = (0..n)
            .map(|_| thread_rng().sample(StandardNormal))
            .collect::<Vec<f64>>();
        let mut acc = CovarianceAccumulator::new(1);
        for val in &v {
            acc.step(::std::iter::once(*val));
        }
        assert!(
            (f64::from(n).sqrt() * acc.mean[0]).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!((1.0 - acc.cov[0]).abs() < 0.1, "{}", acc.cov);
    }

    #[test]
    fn std_normal_uncorr() {
        let n = 100_000;
        let v1 = (0..n)
            .map(|_| thread_rng().sample(StandardNormal))
            .collect::<Vec<f64>>();
        let v2 = (0..n)
            .map(|_| thread_rng().sample(StandardNormal))
            .collect::<Vec<f64>>();
        let mut acc = CovarianceAccumulator::new(2);
        for (val1, val2) in v1.iter().zip(v2.iter()) {
            acc.step(::std::iter::once(*val1).chain(::std::iter::once(*val2)));
        }
        assert!(
            (f64::from(n).sqrt() * acc.mean[0]).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!(
            (f64::from(n).sqrt() * acc.mean[1]).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!((1.0 - acc.cov[0]).abs() < 0.1, "{}", acc.cov);
        assert!(acc.cov[1].abs() < 0.1, "{}", acc.cov);
    }

    #[test]
    fn std_normal_offset() {
        let mu = 10.0;
        let n = 100_000;
        let v = (0..n)
            .map(|_| thread_rng().sample(StandardNormal) + mu)
            .collect::<Vec<f64>>();
        let mut acc = CovarianceAccumulator::new(1);
        for val in &v {
            acc.step(::std::iter::once(*val));
        }
        assert!(
            (f64::from(n).sqrt() * (acc.mean[0] - mu)).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!((1.0 - acc.cov[0]).abs() < 0.1, "{}", acc.cov);
    }

    #[test]
    fn std_normal_uncorr_offset() {
        let mu = 10.0;
        let n = 100_000;
        let v1 = (0..n)
            .map(|_| thread_rng().sample(StandardNormal) + mu)
            .collect::<Vec<f64>>();
        let v2 = (0..n)
            .map(|_| thread_rng().sample(StandardNormal) + mu)
            .collect::<Vec<f64>>();
        let mut acc = CovarianceAccumulator::new(2);
        for (val1, val2) in v1.iter().zip(v2.iter()) {
            acc.step(::std::iter::once(*val1).chain(::std::iter::once(*val2)));
        }
        assert!(
            (f64::from(n).sqrt() * (acc.mean[0] - mu)).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!(
            (f64::from(n).sqrt() * (acc.mean[1] - mu)).abs() < 3.0,
            "{}",
            acc.mean
        );
        assert!((1.0 - acc.cov[0]).abs() < 0.1, "{}", acc.cov);
        assert!(acc.cov[1].abs() < 0.1, "{}", acc.cov);
    }
}
