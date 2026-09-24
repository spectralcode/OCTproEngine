# Curve CSV files

Use **semicolon (`;`) separators**, **dot (`.`) decimals**, and a **header line**. Indices start at **0**.

Let `N` be the configured `signalLength`.

| Curve / profile | CSV header | Meaning of values | Rows for processing |
|---|---|---|---|
| K-linearization / resampling LUT | `index;value` | Input sample position; fractional positions allowed | N |
| Window / apodization | `index;value` | Multiplication factor for each spectral sample | N |
| Dispersion compensation | `index;value` | Phase in radians | N |
| Post-process background | `index;value` | Background level in processed output intensity units | N/2 |
| Fixed pattern noise (FPN) | `index;real;imaginary` | Real and imaginary components of each complex depth sample | N/2 |

**Real curve example:**

```csv
# Window Function
index;value
0;0.5
1;0.75
2;1.0
```

**Fixed pattern noise (FPN) example:**

```csv
# Fixed Pattern Noise Profile
index;real;imaginary
0;0.25;-0.1
1;0.5;0.2
```

**Loading**

- Blank lines and whole-line `#` comments are ignored.
- Values must be finite numbers. Scientific notation is supported.
- Rows are loaded in file order; the index column is ignored.
- Real-curve headers may also use OCTproZ's `Sample Number;Sample Value`.
- Set `signalLength` before loading into `ProcessorConfiguration`. Short curves are padded with zeros; long curves are truncated to the processing lengths above.

**Saving**

Files use the headers shown above. Configuration exports include a `#` title and save the original supplied samples, before padding or truncation. Direct Processor exports save the active profiles.

For OCTproZ versions that do not skip `#` comments, remove those lines before importing.

Background **frames** use raw float32 binary files, not CSV.
