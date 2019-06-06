use errors::*;

use std::fmt;

pub trait RareEvent: fmt::Display {
    /// the last possible time point of the event
    fn max_time(&self) -> f64;

    /// an indicator whether 'state' belongs to the rare event region.
    fn is_rare(&self, time: &TimeInterval, state: &[i32]) -> bool;

    /// distance to the rare event region.
    fn distance(&self, time: &TimeInterval, state: &[i32]) -> i32;
}

#[derive(Debug, Clone, Copy)]
enum ThresholdType {
    Lower,
    Upper,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeInterval(f64, f64);

impl TimeInterval {
    pub fn new_unchecked(lo: f64, hi: f64) -> TimeInterval {
        TimeInterval(lo, hi)
    }

    pub fn new(lo: f64, hi: f64) -> Result<TimeInterval> {
        if hi < lo {
            bail!("Illegal interval bounds (lo={}, hi={})", lo, hi);
        }
        Ok(TimeInterval::new_unchecked(lo, hi))
    }

    pub fn contains(&self, time: f64) -> bool {
        self.0 <= time && time <= self.1
    }

    pub fn intersects(&self, other: &TimeInterval) -> bool {
        if self.0 < other.0 {
            self.1 > other.0
        } else {
            other.1 > self.0
        }
    }
}

impl fmt::Display for TimeInterval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{},{}]", self.0, self.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EventTime {
    Point(f64),
    Interval(TimeInterval),
}

impl EventTime {
    pub fn is_valid(&self, int: &TimeInterval) -> bool {
        match self {
            EventTime::Point(t) => int.contains(*t),
            EventTime::Interval(i) => i.intersects(int),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ThresholdEvent {
    /// time point of the event.
    time: EventTime,

    /// variable index in the state vector.
    pub var_idx: usize,

    /// variable threshold (inclusive)
    pub threshold: i32,

    /// threshold type
    ty: ThresholdType,
}

impl ThresholdEvent {
    pub fn new_lower(time: EventTime, var_idx: usize, threshold: i32) -> ThresholdEvent {
        ThresholdEvent {
            time,
            var_idx,
            threshold,
            ty: ThresholdType::Lower,
        }
    }

    pub fn new_upper(time: EventTime, var_idx: usize, threshold: i32) -> ThresholdEvent {
        ThresholdEvent {
            time,
            var_idx,
            threshold,
            ty: ThresholdType::Upper,
        }
    }
}

impl RareEvent for ThresholdEvent {
    fn max_time(&self) -> f64 {
        match self.time {
            EventTime::Point(t) => t,
            EventTime::Interval(ref i) => i.1,
        }
    }

    fn is_rare(&self, time: &TimeInterval, state: &[i32]) -> bool {
        self.time.is_valid(time)
            && match self.ty {
                ThresholdType::Lower => state[self.var_idx] >= self.threshold,
                ThresholdType::Upper => state[self.var_idx] <= self.threshold,
            }
    }

    fn distance(&self, time: &TimeInterval, state: &[i32]) -> i32 {
        if self.is_rare(time, state) {
            0
        } else {
            (self.threshold - state[self.var_idx]).abs()
        }
    }
}

impl fmt::Display for ThresholdEvent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let rel = match self.ty {
            ThresholdType::Lower => ">=",
            ThresholdType::Upper => "<=",
        };
        let (time, app) = match self.time {
            EventTime::Point(t) => (format!("{}", t), String::new()),
            EventTime::Interval(ref i) => ("t".to_string(), format!(", t in {}", i)),
        };
        write!(
            f,
            "X{}({}) {} {}{}",
            self.var_idx, time, rel, self.threshold, app
        )
    }
}

pub struct CompositeEvent {
    tmax: f64,
    pub events: Vec<ThresholdEvent>,
}

impl CompositeEvent {
    pub fn new(events: Vec<ThresholdEvent>) -> Result<CompositeEvent> {
        let mut tmax: Option<f64> = None;
        for e in &events {
            if let Some(t) = tmax {
                if (t - e.max_time()).abs() > std::f64::EPSILON {
                    bail!(format!("illegal time bounds {} != {}", t, e.max_time()));
                }
            } else {
                tmax = Some(e.max_time());
            }
        }
        if events.is_empty() {
            bail!("no events for composite event");
        }
        Ok(CompositeEvent {
            tmax: tmax.unwrap(),
            events,
        })
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.len() == 0
    }
}

impl RareEvent for CompositeEvent {
    fn max_time(&self) -> f64 {
        self.tmax
    }

    fn is_rare(&self, time: &TimeInterval, state: &[i32]) -> bool {
        self.events.iter().all(|e| e.is_rare(time, state))
    }

    fn distance(&self, time: &TimeInterval, state: &[i32]) -> i32 {
        self.events.iter().map(|e| e.distance(time, state)).sum()
    }
}

impl fmt::Display for CompositeEvent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            self.events
                .iter()
                .map(|e| format!("{}", e))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
