# Gas Sensor data

These files are unmodified copies of batches 6-9 from Gas Sensor Array Drift at Different Concentrations, distributed by the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/270/gas+sensor+array+drift+dataset+at+different+concentrations

Dataset citation: Vergara, A. (2012). Gas Sensor Array Drift at Different Concentrations [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MK6M

The UCI license section identifies the dataset as CC BY 4.0: https://creativecommons.org/licenses/by/4.0/

The source page also contains the original research-use description and requests citation of these papers. Consult that page for the complete dataset documentation and terms.

- Vergara et al., "Chemical gas sensor drift compensation using classifier ensembles."
- Rodriguez-Lujan et al., "On the calibration of sensor arrays for pattern recognition using the minimal number of experiments."

| File | Observations | Features | Role |
|---|---:|---:|---|
| batch6.dat | 2300 | 128 | Source |
| batch7.dat | 3613 | 128 | Source |
| batch8.dat | 294 | 128 | Source |
| batch9.dat | 470 | 128 | Target |

Each line contains a class label, concentration, and indexed feature values. The demo parses these files without modifying them. Concentration is not used as an input feature.
