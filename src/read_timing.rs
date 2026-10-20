//! Optional RISC-V32 software timing, not an electrical waveform measurement.
use core::cell::Cell;
use critical_section::{CriticalSection, Mutex};

/// Generic RISC-V uses mcycle; ESP32-C3 uses its custom MPCCR register.
pub const COUNTER_CSR: u16 = if cfg!(feature = "measure-slot-timing-esp32c3") {
    0x7e2
} else {
    0xb00
};

#[cfg(feature = "measure-slot-timing-esp32c3")]
pub fn esp32c3_cycle_mode_valid(events: u32, mode: u32) -> bool {
    events & 1 != 0 && mode & 3 == 1 // CYCLE selected, enabled, wrapping.
}

/// Read configuration only: HAL/RNG also use this shared counter.
#[cfg(all(feature = "measure-slot-timing-esp32c3", target_arch = "riscv32"))]
pub fn esp32c3_counter_config() -> (u32, u32) {
    critical_section::with(|_| {
        let events: u32;
        let mode: u32;
        unsafe {
            core::arch::asm!(
                "csrr {events}, 0x7e0",
                "csrr {mode}, 0x7e1",
                events = out(reg) events,
                mode = out(reg) mode,
                options(nostack)
            );
        }
        (events, mode)
    })
}

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
static INNER: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));
static PRESENCE: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));
static WRITE_ONE: Mutex<Cell<ReadTiming>> = Mutex::new(Cell::new(ReadTiming::new(u32::MAX)));

/// Reset before acquisition, with a budget based on the actual CPU clock.
pub fn reset(budget_cycles: u32) {
    critical_section::with(|cs| {
        TIMING.borrow(cs).set(ReadTiming::new(budget_cycles));
        INNER.borrow(cs).set(ReadTiming::new(budget_cycles));
    });
}

pub fn snapshot() -> ReadTiming {
    critical_section::with(|cs| TIMING.borrow(cs).get())
}

/// Narrower CPU interval: after driving low until before the data GPIO read.
pub fn inner_snapshot() -> ReadTiming {
    critical_section::with(|cs| INNER.borrow(cs).get())
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
        core::arch::asm!("csrr {value}, {counter}", value = out(reg) value,
            counter = const COUNTER_CSR, options(nostack));
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
pub(crate) fn record_inner(cs: CriticalSection<'_>, started: u32, ended: u32) {
    let mut timing = INNER.borrow(cs).get();
    timing.observe(ended.wrapping_sub(started));
    INNER.borrow(cs).set(timing);
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
    fn counter_selection_and_esp32c3_cycle_mode_are_explicit() {
        assert_eq!(
            COUNTER_CSR,
            if cfg!(feature = "measure-slot-timing-esp32c3") {
                0x7e2
            } else {
                0xb00
            }
        );
        #[cfg(feature = "measure-slot-timing-esp32c3")]
        {
            assert!(esp32c3_cycle_mode_valid(1, 1));
            assert!(esp32c3_cycle_mode_valid(3, 1));
            assert!(!esp32c3_cycle_mode_valid(0, 1));
            assert!(!esp32c3_cycle_mode_valid(2, 1));
            assert!(!esp32c3_cycle_mode_valid(1, 0));
            assert!(!esp32c3_cycle_mode_valid(1, 3));
        }
    }

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
        assert_eq!(inner_snapshot(), ReadTiming::new(2400));
        reset_presence(6000);
        assert_eq!(presence_snapshot(), ReadTiming::new(6000));
        reset_write_one(1200);
        assert_eq!(write_one_snapshot(), ReadTiming::new(1200));
        assert_eq!(presence_snapshot(), ReadTiming::new(6000));
        assert_eq!(snapshot(), ReadTiming::new(2400));
    }
}
