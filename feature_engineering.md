# Feature Engineering Specification (12 → 104)

The same feature engineering is duplicated in each standalone model script so that each can be read, executed and committed independently.

## Raw sensor columns (12)

`Tem`, `Hum`, `Pressure`, `Rain`, `Light`, `Ax`, `Ay`, `Az`, `Wx`, `Wy`, `Wz`, `Moisture`

## Added features (92)

- 6 cyclical timestamp encodings: `time_sin`, `time_cos`, `doy_sin`, `doy_cos`, `weekday_sin`, `weekday_cos`
- 2 vector magnitudes: `accel_magnitude`, `angular_rate_magnitude`
- For each of 12 sensors, 7 transformations:
  - lag 1, lag 2, lag 3
  - rolling mean, window 3
  - rolling standard deviation, window 3
  - rolling mean, window 6
  - first difference

Total: 12 + 6 + 2 + (12 × 7) = **104**.

## Implementation notes

- Rows are parsed and sorted by `Date`.
- The target is numeric and rows with missing Date or target are removed.
- Segment IDs are formed where timestamp differences exceed 30 minutes.
- Lags, rolling statistics and differences are computed within each segment.
- Rolling mean includes the current observation.
- Rolling standard deviation requires at least two values.
- Missing feature values at segment boundaries are filled using medians fit on the training partition.
- The code does not assert the target has a verified fixed-time horizon; target semantics need external documentation.
