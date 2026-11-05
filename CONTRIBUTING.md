# Contributing

Include the hardware, model revision, quantization, engine commit, changed settings and workload behind a result. Keep individual measurements, errors and output lengths. Use public or synthetic inputs.

For code, run `python -B -m unittest discover -s tests -v` and `python -B examples/verify_results.py`. New tests should target an actual format, state or behavior regression.

Keep engine changes and results-only submissions separate. Strata contributions follow its own [contribution rules](https://github.com/Niko1221/Strata/blob/main/CONTRIBUTING.md).

Private prompts, credentials, machine addresses and runtime profiles require deliberate selection before publication. Aggregate expert profiles can reveal workload characteristics.
