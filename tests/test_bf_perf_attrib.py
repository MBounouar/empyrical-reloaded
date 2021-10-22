import pandas as pd
from empyrical.bf_perf_attrib import brinson_fachler_perf_attrib

"""
Examples taken from:
Practical Portfolio Performance Measurement and Attribution.
Brison and Fachler results p.126-129 and p.201 (Frongello)
p. 215 for multi level attribution
"""

dts = pd.date_range("2014-3-31", periods=4, freq="Q")
cntrs = ["UK", "JP", "US"]
mindex = pd.MultiIndex.from_product([dts, cntrs], names=["date", "countries"])

portfolio_weights = pd.Series(
    [0.4, 0.3, 0.3, 0.7, 0.2, 0.1, 0.3, 0.5, 0.2, 0.3, 0.5, 0.2],
    index=mindex,
    name="weights",
)

benchmark_weights = pd.Series(
    [0.4, 0.2, 0.4, 0.4, 0.3, 0.3, 0.5, 0.4, 0.1, 0.4, 0.4, 0.2],
    index=mindex,
    name="weights",
)

portfolio_returns = pd.Series(
    [
        0.2,
        -0.05,
        0.06,
        -0.05,
        0.03,
        -0.05,
        -0.2,
        0.08,
        -0.15,
        0.1,
        -0.07,
        0.25,
    ],
    index=mindex,
    name="returns",
)
benchmark_returns = pd.Series(
    [
        0.1,
        -0.04,
        0.08,
        -0.07,
        0.04,
        -0.1,
        -0.25,
        0.05,
        -0.2,
        0.05,
        -0.05,
        0.1,
    ],
    index=mindex,
    name="returns",
)


class TestBFPerfAttrib:
    def test_bf_perf_attrib_simple(self):
        result = brinson_fachler_perf_attrib(
            pf_pos_returns=portfolio_returns,
            pf_pos_weights=portfolio_weights,
            bm_pos_returns=benchmark_returns,
            bm_pos_weights=benchmark_weights,
            smoothing=True,
        )
