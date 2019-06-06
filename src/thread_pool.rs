use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use errors::*;

enum Message {
    NewJob(Job),
    Terminate,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: mpsc::Sender<Message>,
}

trait FnBox {
    fn call_box(self: Box<Self>) -> Result<()>;
}

impl<F: FnOnce() -> Result<()>> FnBox for F {
    #[allow(unknown_lints, boxed_local)]
    fn call_box(self: Box<F>) -> Result<()> {
        (*self)()
    }
}

type Job = Box<FnBox + Send + 'static>;

impl ThreadPool {
    pub fn new(size: usize) -> Result<ThreadPool> {
        assert!(size > 0);
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            workers.push(Worker::new(id, Arc::clone(&receiver))?);
        }
        Ok(ThreadPool { workers, sender })
    }

    pub fn execute<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        let job = Box::new(f);
        self.sender
            .send(Message::NewJob(job))
            .chain_err(|| "could not execute job")
    }

    pub fn terminate(&mut self) -> Result<()> {
        for _ in &mut self.workers {
            self.sender
                .send(Message::Terminate)
                .chain_err(|| "unable to send termination message")?;
        }
        for mut worker in &mut self.workers {
            let worker_thread = &mut worker.thread;
            if let Some(thread) = worker_thread.take() {
                return match thread.join() {
                    Ok(res) => res,
                    Err(_) => bail!("unable to join thread"),
                };
            }
        }
        Ok(())
    }
}

struct Worker {
    _id: usize,
    thread: Option<thread::JoinHandle<Result<()>>>,
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<mpsc::Receiver<Message>>>) -> Result<Worker> {
        let thread = thread::spawn(move || {
            loop {
                if let Ok(ref mut r) = receiver.lock() {
                    let message = (**r).recv().chain_err(|| "unable to reveive message")?;
                    match message {
                        Message::NewJob(job) => {
                            debug!("Worker {} got a new job", id);
                            job.call_box()?;
                        }
                        Message::Terminate => {
                            break;
                        }
                    }
                } else {
                    bail!("unable to lock receiver");
                }
            }
            Ok(())
        });

        Ok(Worker {
            _id: id,
            thread: Some(thread)
        })
    }
}
