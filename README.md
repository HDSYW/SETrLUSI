# SETrLUSI

## Synthetic and UCI experiment demo

The original demo is organized under [`examples/synthetic_uci/`](examples/synthetic_uci/). Open [`RUN_SYNTHETIC_UCI.m`](examples/synthetic_uci/RUN_SYNTHETIC_UCI.m) in MATLAB and click Run. Its original script, synthetic data, UCI data, and example output are included in that folder.

## Signal-processing experiment demo

Open [`examples/signal_processing/RUN_DEMO.m`](examples/signal_processing/RUN_DEMO.m) in MATLAB and click Run. This example runs SETrLUSI on four Gas Sensor transfer tasks with batch B9 as the target domain and seeds 1-10. Raw batches 6-9 are included.

See the [signal-processing README](examples/signal_processing/README.md) for requirements, parameter settings, timing definitions, checkpoint recovery, and the historical evaluation-label-based parameter-selection protocol. These are not independent-test estimates.

Both demos reuse the shared repository-root `Model/` and `tools/` directories. The signal-processing demo also includes task-specific functions under `examples/signal_processing/support/`. Neither launcher installs additional software.
