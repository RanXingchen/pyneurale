# Changelog

Notable public release changes are recorded in this file.

## Unreleased

Initial public preview. PyNeurale is pre-alpha; public APIs may change before
the first stable release. See [SUPPORT.md](SUPPORT.md) for the current platform
and build support matrix.

### Added

- Typed neural signals, events, trials, features, and spikes, with signal
  processing, model, decoding, and offline sorting APIs.
- Native streaming pipelines with bounded queues, simulated acquisition,
  explicit device selection, and fault handling.
- External acquisition providers through native or Python SDK adapters, plus an
  optional LSL provider for continuous numerical streams. Physical device
  compatibility has not been validated.
- NRF v1 reading, writing, inspection, repair, and replay. `SessionRecorder`
  captures live streams through bounded queues into a continuously appended
  local spool before offline NRF finalization.
- Headless Center-Out, WebGrid, Speech, and SSVEP tasks, with optional desktop
  presentation and input support.
