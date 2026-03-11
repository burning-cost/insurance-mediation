"""
GLM wrappers for outcome and mediator models.

These wrappers standardise how statsmodels GLM results are used for
mediation analysis — specifically the predict() interface which needs
to handle offset terms, design matrices, and counterfactual covariate
values consistently.

Design choice: we wrap statsmodels results objects rather than re-fitting
them. This lets users fit complex models (with interactions, splines,
custom link functions) and hand them to us rather than specifying a
formula string. It also means the outcome model can be a GLMResultsWrapper
from a production pricing model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


OutcomeModelType = Literal["poisson", "gamma", "tweedie", "gaussian"]
MediatorModelType = Literal["linear", "logistic"]


class OutcomeModel:
    """Wrapper around a fitted statsmodels GLM for counterfactual prediction.

    Parameters
    ----------
    model_type : str
        GLM family — "poisson", "gamma", "tweedie", or "gaussian".
    formula : str or None
        Patsy formula string. If provided, the model is fitted internally.
        If None, pass a pre-fitted statsmodels result via fit_result.
    exposure_col : str or None
        Column name for the exposure offset. Used for Poisson and Tweedie
        models where the outcome is a count or burning cost per unit exposure.
    tweedie_var_power : float
        Variance power for Tweedie family. Only used when model_type="tweedie".
        Default 1.5 (compound Poisson-Gamma).
    """

    def __init__(
        self,
        model_type: OutcomeModelType = "poisson",
        formula: str | None = None,
        exposure_col: str | None = None,
        tweedie_var_power: float = 1.5,
    ) -> None:
        self.model_type = model_type
        self.formula = formula
        self.exposure_col = exposure_col
        self.tweedie_var_power = tweedie_var_power
        self._fit_result = None
        self._feature_names: list[str] | None = None

    def fit(self, data: pd.DataFrame, outcome_col: str) -> "OutcomeModel":
        """Fit the GLM outcome model.

        Parameters
        ----------
        data : pd.DataFrame
        outcome_col : str

        Returns
        -------
        self
        """
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        if self.formula is None:
            raise ValueError(
                "formula must be set before calling fit(). "
                "Alternatively use from_fit_result() to wrap a pre-fitted model."
            )

        family = self._get_family()

        if self.exposure_col is not None:
            # Exposure as offset on log scale (log-link models)
            exposure = data[self.exposure_col]
            offset = np.log(exposure.clip(lower=1e-9))
            self._fit_result = smf.glm(
                formula=self.formula,
                data=data,
                family=family,
                offset=offset,
            ).fit(disp=False)
        else:
            self._fit_result = smf.glm(
                formula=self.formula,
                data=data,
                family=family,
            ).fit(disp=False)

        return self

    @classmethod
    def from_fit_result(
        cls,
        fit_result,
        model_type: OutcomeModelType,
        exposure_col: str | None = None,
    ) -> "OutcomeModel":
        """Wrap a pre-fitted statsmodels GLM result.

        Parameters
        ----------
        fit_result : statsmodels GLMResultsWrapper
        model_type : str
        exposure_col : str or None

        Returns
        -------
        OutcomeModel
        """
        obj = cls(model_type=model_type, exposure_col=exposure_col)
        obj._fit_result = fit_result
        return obj

    def _get_family(self):
        """Get the statsmodels family object."""
        import statsmodels.api as sm

        if self.model_type == "poisson":
            return sm.families.Poisson(link=sm.families.links.Log())
        elif self.model_type == "gamma":
            return sm.families.Gamma(link=sm.families.links.Log())
        elif self.model_type == "tweedie":
            return sm.families.Tweedie(
                var_power=self.tweedie_var_power,
                link=sm.families.links.Log(),
            )
        elif self.model_type == "gaussian":
            return sm.families.Gaussian(link=sm.families.links.identity())
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def predict(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        treatment_value,
        mediator_col: str,
        mediator_value: float | None = None,
        use_offset: bool = True,
    ) -> np.ndarray:
        """Predict outcome under a counterfactual assignment.

        Sets treatment to treatment_value and (optionally) mediator to
        mediator_value, then predicts the outcome. All other covariates
        remain at their observed values.

        Parameters
        ----------
        data : pd.DataFrame
            Original data (covariates at observed values).
        treatment_col : str
            Column name of the treatment variable.
        treatment_value : any
            The counterfactual treatment level to assign to all rows.
        mediator_col : str
            Column name of the mediator variable.
        mediator_value : float or None
            If provided, sets all rows' mediator to this value (for CDE).
            If None, leaves the mediator at observed values.
        use_offset : bool
            Whether to apply the exposure offset. Default True.

        Returns
        -------
        np.ndarray of shape (n,)
            Predicted outcome on the response scale (e.g., claim rate per
            unit exposure for Poisson).
        """
        if self._fit_result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Create counterfactual dataset
        cf_data = data.copy()
        cf_data[treatment_col] = treatment_value
        if mediator_value is not None:
            cf_data[mediator_col] = mediator_value

        if use_offset and self.exposure_col is not None:
            offset = np.log(cf_data[self.exposure_col].clip(lower=1e-9))
            return self._fit_result.predict(cf_data, offset=offset)
        else:
            return self._fit_result.predict(cf_data)

    def predict_linear(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        treatment_value,
        mediator_col: str,
        mediator_value: float | None = None,
    ) -> np.ndarray:
        """Predict on the linear predictor (eta) scale.

        For log-link models, this is the log of the predicted mean.
        Used internally for delta-method CIs.

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if self._fit_result is None:
            raise RuntimeError("Model not fitted.")

        cf_data = data.copy()
        cf_data[treatment_col] = treatment_value
        if mediator_value is not None:
            cf_data[mediator_col] = mediator_value

        if self.exposure_col is not None:
            offset = np.log(cf_data[self.exposure_col].clip(lower=1e-9))
            return self._fit_result.predict(cf_data, offset=offset, linear=True)
        return self._fit_result.predict(cf_data, linear=True)

    @property
    def has_log_link(self) -> bool:
        """True for Poisson, Gamma, Tweedie — False for Gaussian."""
        return self.model_type in ("poisson", "gamma", "tweedie")

    @property
    def is_fitted(self) -> bool:
        return self._fit_result is not None


class MediatorModel:
    """Wrapper for the mediator regression model M ~ A + C.

    This model is used to predict the distribution of the mediator
    under different treatment assignments — required for Monte Carlo
    integration in NDE/NIE estimation.

    For a continuous mediator (IMD decile), use model_type="linear".
    For a binary mediator (flood zone indicator), use model_type="logistic".

    Parameters
    ----------
    model_type : str
        "linear" (OLS) or "logistic" (logit GLM).
    formula : str or None
        Patsy formula. If None, use from_fit_result().
    """

    def __init__(
        self,
        model_type: MediatorModelType = "linear",
        formula: str | None = None,
    ) -> None:
        self.model_type = model_type
        self.formula = formula
        self._fit_result = None
        self._residual_std: float | None = None

    def fit(self, data: pd.DataFrame, mediator_col: str) -> "MediatorModel":
        """Fit the mediator model.

        Parameters
        ----------
        data : pd.DataFrame
        mediator_col : str

        Returns
        -------
        self
        """
        import statsmodels.formula.api as smf

        if self.formula is None:
            raise ValueError("formula must be set before calling fit().")

        if self.model_type == "linear":
            self._fit_result = smf.ols(formula=self.formula, data=data).fit()
            self._residual_std = float(np.sqrt(self._fit_result.mse_resid))
        elif self.model_type == "logistic":
            import statsmodels.api as sm
            self._fit_result = smf.glm(
                formula=self.formula,
                data=data,
                family=sm.families.Binomial(),
            ).fit(disp=False)
        else:
            raise ValueError(f"Unknown mediator model_type: {self.model_type}")

        return self

    @classmethod
    def from_fit_result(
        cls,
        fit_result,
        model_type: MediatorModelType,
    ) -> "MediatorModel":
        """Wrap a pre-fitted statsmodels result.

        Parameters
        ----------
        fit_result : statsmodels result object
        model_type : str
        """
        obj = cls(model_type=model_type)
        obj._fit_result = fit_result
        if model_type == "linear":
            obj._residual_std = float(np.sqrt(fit_result.mse_resid))
        return obj

    def predict_mean(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        treatment_value,
    ) -> np.ndarray:
        """Predict E[M | A=treatment_value, C].

        Parameters
        ----------
        data : pd.DataFrame
        treatment_col : str
        treatment_value : any

        Returns
        -------
        np.ndarray of shape (n,)
        """
        if self._fit_result is None:
            raise RuntimeError("MediatorModel not fitted.")
        cf_data = data.copy()
        cf_data[treatment_col] = treatment_value
        return self._fit_result.predict(cf_data)

    def sample(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        treatment_value,
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample mediator values from its conditional distribution.

        For "linear": samples from N(mu, sigma^2) where mu is the fitted
        mean and sigma is the residual standard deviation.
        For "logistic": samples from Bernoulli(p) where p is the fitted
        probability.

        Parameters
        ----------
        data : pd.DataFrame
        treatment_col : str
        treatment_value : any
        n_samples : int
            Number of samples per observation. Returns array of shape
            (n_samples, n_obs).
        rng : np.random.Generator or None
            Random number generator for reproducibility.

        Returns
        -------
        np.ndarray of shape (n_samples, n_obs)
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = self.predict_mean(data, treatment_col, treatment_value)
        n_obs = len(mu)

        if self.model_type == "linear":
            noise = rng.normal(0, self._residual_std, size=(n_samples, n_obs))
            return mu.to_numpy()[np.newaxis, :] + noise
        elif self.model_type == "logistic":
            u = rng.uniform(size=(n_samples, n_obs))
            return (u < mu.to_numpy()[np.newaxis, :]).astype(float)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    @property
    def is_fitted(self) -> bool:
        return self._fit_result is not None

    @property
    def residual_std(self) -> float | None:
        """Residual standard deviation (linear model only)."""
        return self._residual_std


def build_outcome_model(
    model_type: OutcomeModelType,
    data: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    mediator_col: str,
    covariates: list[str],
    exposure_col: str | None = None,
    interaction: bool = True,
    tweedie_var_power: float = 1.5,
) -> OutcomeModel:
    """Build and fit an outcome model from column names.

    Constructs a formula of the form:
        outcome ~ treatment + mediator + treatment:mediator + C1 + C2 + ...

    The treatment:mediator interaction term is included by default because
    exposure-mediator interaction is common in insurance data (e.g.,
    older vehicles in deprived areas have different risk profiles than
    in affluent areas). Detecting and including this interaction is
    important for correct NDE/NIE estimation.

    Parameters
    ----------
    model_type : str
    data : pd.DataFrame
    outcome_col : str
    treatment_col : str
    mediator_col : str
    covariates : list[str]
    exposure_col : str or None
    interaction : bool
        Include treatment × mediator interaction. Default True.
    tweedie_var_power : float

    Returns
    -------
    OutcomeModel (fitted)
    """
    cov_str = " + ".join(covariates) if covariates else ""
    interaction_str = f" + {treatment_col}:{mediator_col}" if interaction else ""
    if cov_str:
        formula = f"{outcome_col} ~ {treatment_col} + {mediator_col}{interaction_str} + {cov_str}"
    else:
        formula = f"{outcome_col} ~ {treatment_col} + {mediator_col}{interaction_str}"

    model = OutcomeModel(
        model_type=model_type,
        formula=formula,
        exposure_col=exposure_col,
        tweedie_var_power=tweedie_var_power,
    )
    model.fit(data, outcome_col)
    return model


def build_mediator_model(
    model_type: MediatorModelType,
    data: pd.DataFrame,
    mediator_col: str,
    treatment_col: str,
    covariates: list[str],
) -> MediatorModel:
    """Build and fit a mediator model from column names.

    Constructs a formula of the form:
        mediator ~ treatment + C1 + C2 + ...

    Parameters
    ----------
    model_type : str
    data : pd.DataFrame
    mediator_col : str
    treatment_col : str
    covariates : list[str]

    Returns
    -------
    MediatorModel (fitted)
    """
    cov_str = " + ".join(covariates) if covariates else ""
    if cov_str:
        formula = f"{mediator_col} ~ {treatment_col} + {cov_str}"
    else:
        formula = f"{mediator_col} ~ {treatment_col}"

    model = MediatorModel(model_type=model_type, formula=formula)
    model.fit(data, mediator_col)
    return model
