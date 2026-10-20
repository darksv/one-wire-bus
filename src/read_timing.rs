//! Optional RISC-V32 software timing, not an electrical waveform measurement.
use core::cell::Cell;
use critical_section::{CriticalSection, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadTiming {
    pub samples: u32,
    pub min_cycles: u32,
    pub max_cycles: u32,
    pub budget_cycles: u32,
    pub over_budget: u32,
}

impl ReadTiming {
    const fn new(budget_cycles: u32) -> Self {
        Self {
            samples: 0,
            min_cycles: 0,
            max_cycles: 0,
            budget_cycles,
            over_budget: 0,
        }
    }

    fn observe(&mut self, cycles: u32) {
        if self.samples == 0 {
            self.min_cycles = cycles;
        } else {
            self.min_cycles = self.min_cycles.min(cycles);
        }
        self.max_cycles = self.max_cycles.max(cycles);
        self.samples = self.samples.saturating_add(1);
        if cycles > self.budget_cycles {
            self.over_budget = self.over_budget.saturating_add(1);
        }
    }
}

static TIMING: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));
static PRESENCE: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));
static WRITE_ONE: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));

/// Reset before acquisition, with a budget based on the actual CPU clock.
pub fn reset(budget_cycles: u32) {
    critical_section::with(|cs| TIMING.borrow(cs).set(ReadTiming::new(budget_cycles)));
}

pub fn snapshot() -> ReadTiming {
    critical_section::with(|cs| TIMING.borrow(cs).get())
}

/// Presence sampling is timed from before reset release to after GPIO read.
pub fn reset_presence(budget_cycles: u32) {
    critical_section::with(|cs| PRESENCE.borrow(cs).set(ReadTiming::new(budget_cycles)));
}

pub fn presence_snapshot() -> ReadTiming {
    critical_section::with(|cs| PRESENCE.borrow(cs).get())
}

/// Write-one low time is bounded by timestamps bracketing both GPIO writes.
pub fn reset_write_one(budget_cycles: u32) {
    critical_section::with(|cs| WRITE_ONE.borrow(cs).set(ReadTiming::new(budget_cycles)));
}

pub fn write_one_snapshot() -> ReadTiming {
    critical_section::with(|cs| WRITE_ONE.borrow(cs).get())
}

#[cfg(target_arch = "riscv32")]
#[inline(always)]
pub(crate) fn cycles() -> u32 {
    let value: u32;
    // Default memory clobber deliberately keeps the timestamp ordered with
    // respect to the volatile GPIO access; do not mark this asm `nomem`/`pure`.
    unsafe {
        core::arch::asm!("csrr {value}, mcycle", value = out(reg) value, options(nostack));
    }
    value
}

#[cfg(target_arch = "riscv32")]
pub(crate) fn record(cs: CriticalSection<'_>, started: u32, ended: u32) {
    let mut timing = TIMING.borrow(cs).get();
    timing.observe(ended.wrapping_sub(started));
    TIMING.borrow(cs).set(timing);
}

#[cfg(target_arch = "riscv32")]
pub(crate) fn record_presence(cs: CriticalSection<'_>, started: u32, ended: u32) {
    let mut timing = PRESENCE.borrow(cs).get();
    timing.observe(ended.wrapping_sub(started));
    PRESENCE.borrow(cs).set(timing);
}

#[cfg(target_arch = "riscv32")]
pub(crate) fn record_write_one(cs: CriticalSection<'_>, started: u32, ended: u32) {
    let mut timing = WRITE_ONE.borrow(cs).get();
    timing.observe(ended.wrapping_sub(started));
    WRITE_ONE.borrow(cs).set(timing);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_extrema_and_only_counts_samples_above_the_budget() {
        let mut timing = ReadTiming::new(1200);
        for sample in [900, 1200, 1201, 800] {
            timing.observe(sample);
        }
        assert_eq!(timing.samples, 4);
        assert_eq!((timing.min_cycles, timing.max_cycles), (800, 1201));
        assert_eq!(timing.over_budget, 1);
    }

    #[test]
    fn counter_wrap_is_a_short_interval_and_telemetry_counters_saturate() {
        let elapsed = 20_u32.wrapping_sub(u32::MAX - 9);
        assert_eq!(elapsed, 30);
        let mut timing = ReadTiming::new(10);
        timing.samples = u32::MAX;
        timing.over_budget = u32::MAX;
        timing.min_cycles = 1;
        timing.observe(elapsed);
        assert_eq!(timing.samples, u32::MAX);
        assert_eq!(timing.over_budget, u32::MAX);
        assert_eq!(timing.max_cycles, 30);
    }

    #[test]
    fn reset_discards_old_measurements_and_installs_the_new_budget() {
        reset(1200);
        assert_eq!(snapshot(), ReadTiming::new(1200));
        reset(2400);
        assert_eq!(snapshot(), ReadTiming::new(2400));
        reset_presence(6000);
        assert_eq!(presence_snapshot(), ReadTiming::new(6000));
        reset_write_one(1200);
        assert_eq!(write_one_snapshot(), ReadTiming::new(1200));
        assert_eq!(presence_snapshot(), ReadTiming::new(6000));
        assert_eq!(snapshot(), ReadTiming::new(2400));
    }
}
