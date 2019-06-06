extern crate indicatif;

use self::indicatif::{ProgressBar, ProgressStyle};

use errors::*;

pub struct Progress(ProgressBar);

impl Drop for Progress {
    fn drop(&mut self) {
        let &mut Progress(ref pb) = self;
        pb.finish_and_clear();
    }
}

impl Progress {
    pub fn new(num_events: u64) -> Result<Progress> {
        let pb = ProgressBar::new(num_events);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:50}] {pos}/{len} ({eta})")
                .progress_chars("=> "),
        );
        Ok(Progress(pb))
    }

    pub fn tick(&mut self) {
        self.0.inc(1);
    }
}
