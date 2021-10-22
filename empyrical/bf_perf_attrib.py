import pandas as pd
from collections import OrderedDict
from typing import Optional


def brinson_fachler_perf_attrib(
    pf_pos_returns: pd.Series,
    pf_pos_weights: pd.Series,
    bm_pos_returns: pd.Series,
    bm_pos_weights: pd.Series,
    pf_returns: Optional[pd.Series] = None,
    bm_returns: Optional[pd.Series] = None,
    geometric: bool = False,
    smoothing: bool = True,
) -> pd.DataFrame:
    """
    Brinson and Fachler type attribution.

    Interaction is included with selection.
    For Arithmetic:
    allocation = (pf_wi - bmk_wi) * (Rbmk_i - Rbmk)
    selection = pf_wi * (Rpf_i - Rbmk_i)

    For Geometric:
    allocation = (pf_wi - bmk_wi) * (1 + Rbmk_i)/(1 + Rbmk) - 1
    selection = pf_wi * ((1 + Rpf_i)/(1 + Rbmk_i) - 1) * (1 + Rbmk_i)/(1 + Rs)

    if geometric is set to True smoothing algorithm is ignored

    Parameters
    ----------

    Returns
    -------
    brinson_fachler_perf_attrib:
    """
    active_pos_weights = pf_pos_weights - bm_pos_weights

    if bm_returns is None:
        bm_returns = (bm_pos_weights * bm_pos_returns).groupby(level=0).sum()

    attr_df = OrderedDict()

    if geometric:
        active_pos_returns = ((1 + pf_pos_returns) / (1 + bm_pos_returns)) - 1
        s_notional_return = (
            (pf_pos_weights * bm_pos_returns).groupby(level=0).sum()
        )
        attr_df["allocation"] = active_pos_weights * (
            (1 + bm_pos_returns).div(1 + bm_returns, level=0) - 1.0
        )
        # adjusted geometric selection see: page 206
        attr_df["selection"] = (
            pf_pos_weights
            * active_pos_returns
            * (1 + bm_pos_returns)
            / (1 + s_notional_return)
        )
    else:
        active_pos_returns = pf_pos_returns - bm_pos_returns
        attr_df["allocation"] = active_pos_weights * bm_pos_returns.add(
            -bm_returns, level=0
        )
        attr_df["selection"] = pf_pos_weights * active_pos_returns

    attr_df = pd.concat(attr_df, axis=1)

    if not geometric and smoothing:
        if pf_returns is None:
            pf_returns = (
                (pf_pos_weights * pf_pos_returns).groupby(level=0).sum()
            )
        attr_df = frongello_smoothing(attr_df, pf_returns, bm_returns)

    return attr_df


def frongello_smoothing(
    performance_attr: pd.DataFrame,
    pf_returns: pd.Series,
    bm_returns: pd.Series,
) -> pd.DataFrame:
    dts = performance_attr.index.unique(0)
    a = (1 + pf_returns).cumprod()
    for i, dt in enumerate(dts[1:]):
        performance_attr.loc[dt] = (
            performance_attr.loc[dt] * a[dts[i]]
            + bm_returns[dt]
            * performance_attr.loc[: dts[i], :]
            .groupby(level=-1, sort=False)
            .sum()
        ).values

    return performance_attr


def mlvl_brinson_fachler_perf_attrib(
    pf_pos_returns: pd.Series,
    pf_pos_weights: pd.Series,
    bm_pos_returns: pd.Series,
    bm_pos_weights: pd.Series,
) -> pd.DataFrame:
    """
    Brinson and Fachler multilevel type attribution.

    Parameters
    ----------

    Returns
    -------
    multilevel brinson_fachler_perf_attrib:
    """
    nlvls = len(pf_pos_returns.index.levshape)
    mlvls = [list(range(x)) for x in range(2, nlvls + 1)]

    lvl_names = pf_pos_returns.index.names
    if len(set(lvl_names)) < nlvls:
        lvl_names = ["date"] + [f"lvl{x}" for x in range(1, nlvls)]

    bm_returns = (bm_pos_weights * bm_pos_returns).groupby(level=0).sum()

    attr_df = []
    s_adj = [1]
    bm_lvl_rets = [bm_returns]

    for i, mlvl in enumerate(mlvls):
        pf_pos_lvl_weights = pf_pos_weights.groupby(
            level=mlvl, sort=False
        ).sum()

        bm_pos_lvl_weights = bm_pos_weights.groupby(
            level=mlvl, sort=False
        ).sum()

        bm_lvl_ret = (bm_pos_weights * bm_pos_returns).groupby(
            level=mlvl, sort=False
        ).sum() / bm_pos_lvl_weights

        bm_lvl_rets.append(bm_lvl_ret)
        active_lvl_weights = pf_pos_lvl_weights - bm_pos_lvl_weights

        allocation = (
            (
                active_lvl_weights
                * ((1 + bm_lvl_ret) / (1 + bm_lvl_rets[i]) - 1.0)
                * s_adj[i]
            )
            .rename(f"Allocation {lvl_names[i+1]}")
            .to_frame()
        )

        if len(allocation.index.names) < nlvls:
            mkeys = lvl_names[len(allocation.index.names) :]
            allocation = allocation.assign(**{x: "" for x in mkeys}).set_index(
                mkeys, append=True
            )

        attr_df.append(allocation)

        s_notional_return = sum(pf_pos_lvl_weights * bm_lvl_ret)
        s_adj.append((1 + bm_lvl_ret) / (1 + s_notional_return))

    # adjusted geometric selection see: page 206
    selection = (
        (
            pf_pos_weights
            * ((1 + pf_pos_returns) / (1 + bm_pos_returns) - 1)
            * (1 + bm_pos_returns)
            / (1 + s_notional_return)
        )
        .rename("Selection")
        .to_frame()
    )
    attr_df.append(selection)

    attr_df = pd.concat(attr_df, axis=1)
    return attr_df


if __name__ == "__main__":
    """
    Examples taken from:
    Practical Portfolio Performance Measurement and Attribution.
    Brison and Fachler results p.126-129 and p.201 (Frongello)
    p. 215 for multi level attribution
    """
    dts = pd.date_range("2014-3-31", periods=4, freq="Q")
    cntrs = ["UK", "JP", "US"]
    mindex = pd.MultiIndex.from_product(
        [dts, cntrs], names=["date", "countries"]
    )

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
    res_df = brinson_fachler_perf_attrib(
        pf_pos_returns=portfolio_returns,
        pf_pos_weights=portfolio_weights,
        bm_pos_returns=benchmark_returns,
        bm_pos_weights=benchmark_weights,
        geometric=True,
        # smoothing=True,
    )

    print(res_df.to_string())
    # mpa = MultiLevelPerformanceAttribution(mw1, mr1, mwr1)
    # print(numpy.round((mpa.get_allocation_selection() * 100), 2).to_string())

    # print(mw1.to_string())
