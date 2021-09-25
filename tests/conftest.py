import pytest
import pandas as pd
import numpy as np

rand = np.random.RandomState(1337)


@pytest.fixture(scope="function")
def set_helpers(request):
    request.cls.ser_length = 120
    request.cls.window = 12

    request.cls.returns = pd.Series(
        rand.randn(1, 120)[0] / 100.0,
        index=pd.date_range("2000-1-30", periods=120, freq="M"),
    )

    request.cls.factor_returns = pd.Series(
        rand.randn(1, 120)[0] / 100.0,
        index=pd.date_range("2000-1-30", periods=120, freq="M"),
    )


@pytest.fixture(scope="session")
def returns_types():
    return {
        # Simple benchmark, no drawdown
        "simple_benchmark": pd.Series(
            np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        ),
        # All positive returns, small variance
        "positive_returns": pd.Series(
            np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        ),
        # All negative returns
        "negative_returns": pd.Series(
            np.array([0.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0])
            / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        ),
        # All negative returns
        "all_negative_returns": pd.Series(
            np.array([-2.0, -6.0, -7.0, -1.0, -9.0, -2.0, -6.0, -8.0, -5.0])
            / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        ),
        # Positive and negative returns with max drawdown
        "mixed_returns": pd.Series(
            np.array([np.nan, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0])
            / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="D"),
        ),
        # Flat line
        "flat_line_1": flat_line_1_tz,
        # Weekly returns
        "weekly_returns": pd.Series(
            np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="W"),
        ),
        # Monthly returns
        "monthly_returns": pd.Series(
            np.array([0.0, 1.0, 10.0, -4.0, 2.0, 3.0, 2.0, 1.0, -10.0]) / 100,
            index=pd.date_range("2000-1-30", periods=9, freq="M"),
        ),
        # Series of length 1
        "one_return": pd.Series(
            np.array([1.0]) / 100,
            index=pd.date_range("2000-1-30", periods=1, freq="D"),
        ),
        # Empty series
        "empty_returns": pd.Series(
            np.array([]) / 100,
            index=pd.date_range("2000-1-30", periods=0, freq="D"),
        ),
        # Random noise
        "noise": noise,
        "noise_uniform": pd.Series(
            rand.uniform(-0.01, 0.01, 1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
        # Random noise inv
        "inv_noise": inv_noise,
        # Flat line
        "flat_line_0": pd.Series(
            np.linspace(0, 0, num=1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
        # Flat line
        "flat_line_1_tz": pd.Series(
            np.linspace(0.01, 0.01, num=1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
        # Positive line
        "pos_line": pd.Series(
            np.linspace(0, 1, num=1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
        # Negative line
        "neg_line": pd.Series(
            np.linspace(0, -1, num=1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
        # Sparse noise, same as noise but with np.nan sprinkled in
        "sparse_noise": sparse_noise,
        # Sparse flat line at 0.01
        "sparse_flat_line_1_tz": sparse_flat_line_1_tz,
        "one": one,
        "two": two,
        "df_index_simple": df_index_simple,
        "df_index_week": df_index_week,
        "df_index_month": df_index_month,
        "df_simple": df_simple,
        "df_week": df_week,
        "df_month": df_month,
    }


@pytest.fixture
def noise():
    # Random noise
    return (
        pd.Series(
            rand.normal(0, 0.001, 1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
    )


@pytest.fixture
def inv_noise():
    return noise.multiply(-1)


@pytest.fixture
def sparse_noise():
    # Sparse noise, same as noise but with np.nan sprinkled in
    replace_nan = rand.choice(noise.index.tolist(), rand.randint(1, 10))
    return noise.replace(replace_nan, np.nan)


@pytest.fixture
def flat_line_1_tz():
    # Flat line
    return (
        pd.Series(
            np.linspace(0.01, 0.01, num=1000),
            index=pd.date_range("2000-1-30", periods=1000, freq="D", tz="UTC"),
        ),
    )


@pytest.fixture
def sparse_flat_line_1_tz():
    # Sparse flat line at 0.01
    replace_nan = rand.choice(noise.index.tolist(), rand.randint(1, 10))
    return flat_line_1_tz.replace(replace_nan, np.nan)


@pytest.fixture
def one():
    return [
        -0.00171614,
        0.01322056,
        0.03063862,
        -0.01422057,
        -0.00489779,
        0.01268925,
        -0.03357711,
        0.01797036,
    ]


@pytest.fixture
def two():
    return (
        [
            0.01846232,
            0.00793951,
            -0.01448395,
            0.00422537,
            -0.00339611,
            0.03756813,
            0.0151531,
            0.03549769,
        ],
    )


@pytest.fixture
def df_index_simple():
    return (pd.date_range("2000-1-30", periods=8, freq="D"),)


@pytest.fixture
def df_index_week():
    return (pd.date_range("2000-1-30", periods=8, freq="W"),)


@pytest.fixture
def df_index_month():
    return (pd.date_range("2000-1-30", periods=8, freq="M"),)


@pytest.fixture
def df_week():
    return pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_week),
            "two": pd.Series(two, index=df_index_week),
        }
    )


@pytest.fixture
def df_month(one, two):
    return pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_month),
            "two": pd.Series(two, index=df_index_month),
        }
    )


@pytest.fixture
def df_simple():
    return pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_simple),
            "two": pd.Series(two, index=df_index_simple),
        }
    )


@pytest.fixture
def df_week():
    return pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_week),
            "two": pd.Series(two, index=df_index_week),
        }
    )


@pytest.fixture
def df_month():
    return pd.DataFrame(
        {
            "one": pd.Series(one, index=df_index_month),
            "two": pd.Series(two, index=df_index_month),
        }
    )


@pytest.fixture
def returns(returns_types, request):
    """Create a Sushi instance based on recipes."""
    name = request.param
    return returns_types[name]
