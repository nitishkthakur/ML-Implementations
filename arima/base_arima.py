# import pandas as pd
# import numpy as np


class ARIMAForecaster:
    """
    A flexible SARIMAX/ARIMAX forecasting class.

    - Supports fixed orders or range search via pmdarima.auto_arima.
    - Optional seasonality with period m.
    - Includes summary, prediction, and diagnostics methods.
    """

    def __init__(self, y, exog=None):
        """
        Parameters
        ----------
        y : pandas Series
            Endogenous time series, indexed by a proper DatetimeIndex.
        exog : pandas Series or DataFrame, optional
            Exogenous regressors aligned with y.
        """
        self.y = y
        self.exog = exog
        self.model = None
        self.res = None

    def fit(self, p, d, q, P=None, D=None, Q=None, m=1):
        """
        Fit SARIMAX or auto_arima depending on parameter types.

        Parameters
        ----------
        p, d, q : int or tuple
            AR, differencing, and MA orders (or (min, max) ranges for auto_arima).
        P, D, Q : int, tuple, or None
            Seasonal AR, differencing, and MA orders (or (min, max)).
            If all are None, no seasonality is used.
        m : int
            Seasonal period (e.g., 12 for monthly data).

        Returns
        -------
        res : fitted model results
        """
        # Determine if we run auto_arima (any tuple) or fixed SARIMAX
        is_range_search = any(isinstance(x, tuple) for x in (p, d, q, P, D, Q))
        seasonal = P is not None and D is not None and Q is not None

        if is_range_search:
            import pmdarima as pm

            # Unpack ranges or fix to (val, val)
            def _range(x):
                if isinstance(x, tuple):
                    return x
                if x is None:
                    return (0, 0)
                return (x, x)

            p_min, p_max = _range(p)
            d_min, d_max = _range(d)
            q_min, q_max = _range(q)
            P_min, P_max = _range(P)
            D_min, D_max = _range(D)
            Q_min, Q_max = _range(Q)

            self.model = pm.auto_arima(
                self.y,
                exogenous=self.exog,
                start_p=p_min,
                max_p=p_max,
                start_q=q_min,
                max_q=q_max,
                d=d_min,
                seasonal=seasonal,
                m=m,
                start_P=P_min,
                max_P=P_max,
                D=D_min,
                start_Q=Q_min,
                max_Q=Q_max,
                trace=True,
                error_action="ignore",
                suppress_warnings=True,
            )
            self.res = self.model

        else:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            order = (p, d, q)
            if seasonal:
                seasonal_order = (P, D, Q, m)
            else:
                seasonal_order = (0, 0, 0, 0)

            self.model = SARIMAX(
                endog=self.y,
                exog=self.exog,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.res = self.model.fit(disp=False)

        return self.res

    def report(self):
        """
        Print a summary report of the fitted model.
        """
        if hasattr(self.res, "summary"):
            print(self.res.summary())
        else:
            print(self.res)

    def predict(self, n_periods, exog_future=None):
        """
        Generate forecasts for the next n_periods.

        If differencing was applied, results are returned on original scale.

        Parameters
        ----------
        n_periods : int
            Number of steps ahead to forecast.
        exog_future : array-like, optional
            Future values of exogenous variables for forecasting.

        Returns
        -------
        forecast : pandas Series or numpy array
            Point forecasts (and confidence intervals if available).
        """
        if exog_future is not None:
            self.exog_future = exog_future

        # statsmodels SARIMAX
        if hasattr(self.res, "get_forecast"):
            fc = self.res.get_forecast(
                steps=n_periods, exog=getattr(self, "exog_future", None)
            )
            return fc.predicted_mean, fc.conf_int()

        # pmdarima auto_arima
        preds = self.res.predict(
            n_periods=n_periods, exogenous=getattr(self, "exog_future", None)
        )
        return preds

    def diagnostics(self):
        """
        Plot residual diagnostics for the fitted model.
        """
        if hasattr(self.res, "plot_diagnostics"):
            self.res.plot_diagnostics(figsize=(12, 8))
        else:
            print("No diagnostics available for this model type.")
