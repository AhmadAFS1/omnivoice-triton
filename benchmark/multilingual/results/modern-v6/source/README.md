`audit.py` is the exact CPU audit source recorded by the adjacent result summary.
It expects the original retained task layout and is preserved for inspection;
it is not a public harness entry point. Raw full-run WAVs are external to this
compact result package. The parameterized public preparation/scoring tool is
`quality/report.py` at the benchmark root. No model or GPU work occurs in this
audit script.
