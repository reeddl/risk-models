"""riskmodels: a collection of portfolio risk models on one shared core.

Subpackages
-----------
core          Data contracts, point-in-time helpers, cross-sectional preprocessing,
              covariance estimators and a synthetic data generator. Shared by every model.
factor_model  Barra/Axioma-style fundamental equity factor model.
position_sizing
              Conviction scores to capped, beta- and vol-targeted weights, using the
              factor model's risk forecast.
"""

__version__ = "0.1.0"
