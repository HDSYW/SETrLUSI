# SETrLUSI

## Original demo

Run `Running_SETrLUSI.m` for the original examples. The original script, datasets, models, and tools are unchanged by the Gas Sensor supplement.

## Gas Sensor B9 supplement

Open [`examples/gas_b9/RUN_DEMO.m`](examples/gas_b9/RUN_DEMO.m) in MATLAB and click Run. This example runs SETrLUSI on four Gas Sensor transfer tasks with batch B9 as the target domain and seeds 1-10. Raw batches 6-9 are included.

See the [supplement README](examples/gas_b9/README.md) for requirements, parameter settings, timing definitions, checkpoint recovery, and the historical evaluation-label-based parameter-selection protocol. These are not independent-test estimates.

The supplement includes task-specific functions under `examples/gas_b9/support/` and reuses existing repository tools and Windows MEX binaries. It does not change the original demo or install additional software.
