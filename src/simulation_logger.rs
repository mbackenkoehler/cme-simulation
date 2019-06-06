use std::cell::RefCell;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::sync::{Arc, Mutex};

use errors::*;
use progressbar::Progress;
use simulation::SimulationLogger;

pub struct CsvSimulationLogger {
    writer: Option<Arc<Mutex<RefCell<BufWriter<File>>>>>,
    progress: Option<Arc<Mutex<Progress>>>,
    write_time: bool,
    write_id: bool,
    names: Vec<String>,
}

impl CsvSimulationLogger {
    pub fn new(
        names: &[String],
        file: Option<String>,
        write_time: bool,
        write_id: bool,
        total_calls: u64,
    ) -> Result<CsvSimulationLogger> {
        let pb = Progress::new(total_calls)?;
        let progress = if file.is_some() {
            Some(Arc::new(Mutex::new(pb)))
        } else {
            None
        };
        let logger = CsvSimulationLogger {
            writer: if let Some(out_file) = file {
                let out = File::create(out_file).chain_err(|| "could not create output file")?;
                Some(Arc::new(Mutex::new(RefCell::new(BufWriter::new(out)))))
            } else {
                None
            },
            progress,
            write_time,
            write_id,
            names: names.to_vec(),
        };
        let mut header = String::from("");
        if write_id {
            header.push_str("id,");
        }
        if write_time {
            header.push_str("time,");
        }
        header.push_str(&names.join(","));
        header.push_str("\n");
        logger
            .write(&header)
            .chain_err(|| "unable to write header")?;
        Ok(logger)
    }

    fn write(&self, s: &str) -> Result<()> {
        let s = s.as_bytes();
        if let Some(ref file_cell) = self.writer {
            let file_cell = &*file_cell.lock().unwrap();
            file_cell
                .borrow_mut()
                .write_all(s)
                .chain_err(|| "could not write to file")?;
        }
        Ok(())
    }
}

impl SimulationLogger for CsvSimulationLogger {
    fn log(&self, id: usize, time: f64, state: &[i32]) -> Result<()> {
        if self.writer.is_some() {
            let state_str = format!("{:?}", state);
            let sstr = state_str
                .get(1..(state_str.len() - 1))
                .chain_err(|| format!("unexpected state str: {:?}", state_str))?;
            let mut row = if self.write_time {
                format!("{},{}\n", time, sstr)
            } else {
                String::from(sstr) + "\n"
            };
            if self.write_id {
                row = format!("{},{}", id, row);
            }
            self.write(&row).chain_err(|| "unable to write row")?;
        } else {
            let state = self
                .names
                .iter()
                .zip(state.iter())
                .map(|(name, count)| format!("{}={:<7}", name, count))
                .collect::<Vec<_>>()
                .join(" ");
            info!("t={:<10.2} {}", time, state);
        };
        Ok(())
    }

    fn job_done(&self, _id: usize, _state: &[i32]) -> Result<()> {
        if let Some(ref progress_cell) = self.progress {
            match progress_cell.try_lock() {
                Ok(ref mut pb) => {
                    pb.tick();
                }
                Err(_) => bail!("could not lock progress bar"),
            }
        }
        Ok(())
    }
}

impl Clone for CsvSimulationLogger {
    fn clone(&self) -> CsvSimulationLogger {
        CsvSimulationLogger {
            writer: self.writer.as_ref().map(Arc::clone),
            progress: self.progress.as_ref().cloned(),
            names: self.names.clone(),
            ..*self
        }
    }
}
