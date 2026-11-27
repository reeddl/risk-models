import pytest

from riskmodels.core.synthetic import generate_market_data
from riskmodels.factor_model import FactorModel, FactorModelConfig


@pytest.fixture(scope="session")
def synthetic():
    return generate_market_data(n_assets=200, n_days=460, seed=7)


@pytest.fixture(scope="session")
def daily_fit(synthetic):
    data, _ = synthetic
    return FactorModel(FactorModelConfig.daily()).fit(data)


@pytest.fixture(scope="session")
def weekly_fit(synthetic):
    data, _ = synthetic
    return FactorModel(FactorModelConfig.weekly()).fit(data)
