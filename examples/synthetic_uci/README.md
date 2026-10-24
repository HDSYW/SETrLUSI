# Synthetic and UCI experiment demo

This folder contains the original SETrLUSI demo for artificially generated synthetic data and UCI benchmark datasets.

## Run

Open `RUN_SYNTHETIC_UCI.m` in MATLAB and click Run. The launcher adds the shared repository models and tools, then runs the original script from this folder so its relative data and result paths remain valid. The previous MATLAB path is restored when the launcher exits.

The original `Running_SETrLUSI.m` is retained without changes. Its default dataset is `rice123`. Dataset selection is controlled by `pa.DA` in that script. Available synthetic datasets are in the `Data/SD*` subdirectories. Available UCI datasets include the Rice, Waveform, and TW examples under `Data/`.

## Files

- `Running_SETrLUSI.m`: original experiment script and settings.
- `Data/`: original synthetic and UCI data files.
- `AutoResult/`: original example output and the destination used by the original result writer.
- `RUN_SYNTHETIC_UCI.m`: launcher for the reorganized repository layout.

The original parameter search, random-seed handling, model selection, and reporting behavior are unchanged. Running this demo performs training and parameter search.

## Shared dependencies

The repository-root `Model/` and `tools/` directories are shared dependencies. Keep this folder inside the repository. The signal-processing experiments are provided separately in `../signal_processing/`.
