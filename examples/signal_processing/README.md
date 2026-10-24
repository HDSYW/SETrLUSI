# Signal-processing experiment demo

This demo runs the existing SETrLUSI implementation using the historical experimental protocol. It does not read accuracy values from spreadsheets or adjust predictions to match manuscript tables. Exact agreement with the manuscript results has not been verified.

## Run

Open `RUN_DEMO.m` in MATLAB and click Run.

Alternatively, set the MATLAB current folder to this directory and execute:

```matlab
RUN_DEMO
```

Run the same entry point again to resume an interrupted experiment. Saved seeds are skipped after checking that the configuration, code hashes, and data hashes match.

To check dependencies and configuration without training or writing experimental results:

```matlab
Demo_SETrLUSI_B9('check')
```

## Configuration

- Tasks: B6 to B9, B6+B7 to B9, B7+B8 to B9, and B6+B7+B8 to B9.
- Method: SETrLUSI only, with seeds 1-10 and 40 runs in total.
- Target partition: class-stratified sampling of B9, with 50 labeled training observations and 420 evaluation observations.
- Model: 128 features, six classes, a linear kernel, H=30, V=I, and the existing solver.
- Per-class parameter grid: RD={0.3,0.5,0.7,0.9}, p3={0.1,0.5,0.9}, and p1={2^-8,2^-4,2^0}. Random auxiliary SVM settings and all ten predicates follow the existing implementation.
- Parameter selection: the original code uses evaluation labels to select per-class parameters and ensemble rounds. These outputs follow the historical selection procedure and are not unbiased estimates on an independent test set.
- Timing: `SearchRunSeconds` includes parameter search, training, prediction, and evaluation within the model call. It excludes data loading and result persistence. It is not the separately measured fixed-parameter `fit_s` reported in the manuscript.

## Bundled data

The `data/` directory contains unmodified `batch6.dat`, `batch7.dat`, `batch8.dat`, and `batch9.dat` files from the project's Gas Sensor Array Drift Dataset at Different Concentrations. Batches 6, 7, and 8 provide the source domains. Batch 9 provides the target domain. The demo reads these bundled files directly and does not use the original external data directory.

The four batches contain 2,300, 3,613, 294, and 470 observations, respectively. Each observation has 128 features. The `.dat` format records the class label, concentration, and indexed feature values. File contents are unchanged, and SHA-256 hashes are checked for checkpoint compatibility.

## Results

Outputs are saved under `results/` in this directory, separately from previous experiments:

- `settings.mat`: configuration and SHA-256 hashes of model dependencies and raw batches 6-9.
- Per-task `seed_01.mat` through `seed_10.mat`: predictions, confusion matrices, selected parameters, standardized target training and evaluation partitions, the RNG state before the model call, and performance metrics.
- Per-task `per_seed.csv` and `summary.csv`.
- Combined `all_seeds.csv` and `summary.csv`, updated after each completed task.

Files are written to temporary paths and read back for verification before being moved to their destination. Existing checkpoints cannot be reused after changing the configuration, code, or raw data. Preserve previous results and select a new output directory for a new experimental configuration.

## Requirements

The data are bundled. This supplementary demo reuses the repository model binaries and common tools, with Gas-specific functions in `support/`. Keep the repository directory layout unchanged.

Required software includes MATLAB (the existing project uses R2021b), Statistics and Machine Learning Toolbox, and the project's LIBSVM and LIBLINEAR MEX binaries. The demo does not install dependencies, invoke Python, or run other baselines.

## Repository integration

The original synthetic and UCI experiments are organized separately in `../synthetic_uci/`. Their experiment script and the shared model/tool files are unchanged. Gas-specific model, solver, multiclass split, predicate, and loading functions are isolated in this folder's `support/` directory. These functions are placed first on the MATLAB path only while the demo runs. The previous MATLAB path is restored on exit. Existing LIBSVM and LIBLINEAR MEX binaries are reused.

See `data/README.md` for dataset attribution and citations.
