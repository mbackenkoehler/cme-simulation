/// def omega2(dim, cap):
///     if dim == 1:
///         yield [cap]
///     else:
///         for x in range(cap + 1):
///             for prefix in omega2(dim - 1, cap - x):
///                  yield [x] + prefix
#[derive(Debug)]
pub struct Omega {
    dim: u64,
    cap: u64,
    x: u64,
    prefix: Option<Box<Omega>>,
}

impl Omega {
    pub fn new(dim: u64, cap: u64) -> Omega {
        let prefix = if dim == 1 {
            None
        } else {
            Some(Box::new(Omega::new(dim - 1, cap)))
        };
        let res = Omega {
            dim,
            cap,
            x: 0,
            prefix,
        };
        assert!(!res.done());
        res
    }

    pub fn step(&mut self) {
        if self.dim == 1 {
            self.x = self.cap + 1;
        } else {
            self.prefix.as_mut().unwrap().step();
            if self.prefix.as_ref().map(|p| p.done()).unwrap_or(false) {
                self.x += 1;
                if self.x <= self.cap {
                    self.prefix = Some(Box::new(Omega::new(self.dim - 1, self.cap - self.x)));
                }
            }
        }
    }

    pub fn set_val(&self, tgt: &mut [u64]) {
        if self.dim == 1 {
            tgt[0] = self.cap;
        } else {
            self.prefix.as_ref().unwrap().set_val(tgt);
            tgt[(self.dim - 1) as usize] = self.x;
        }
    }

    pub fn done(&self) -> bool {
        self.x > self.cap
    }
}
