# Public data policy

This portfolio repository contains source code, public weather-derived inputs,
planning scenarios, and synthetic campus-load fixtures.

It intentionally excludes:

- raw SQL exports and cumulative meter readings;
- meter IDs, room or building addresses, remarks, and reviewed labels;
- per-meter interval data and data-quality investigation workbooks;
- API keys, local environment files, sessions, traces, and generated outputs;
- archived project materials whose redistribution status is unclear.

The public load curve is deterministic synthetic data. It preserves the shape
and units required to exercise the optimization and Agent workflow, but it is
not a transformed or sampled copy of the private meter-level dataset.

All dispatch results remain simulation-only and non-executable.
