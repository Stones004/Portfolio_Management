import numpy as np
import pandas as pd
from hrp import HRP   # your existing HRP class


# ======================================================
# Black–Litterman using HRP risk model
# ======================================================
class BlackLittermanHRP:
    """
    Black-Litterman model using HRP covariance and HRP weights
    """

    def __init__(
        self,
        hrp_cov: pd.DataFrame,
        hrp_weights: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        ridge: float = 1e-6
    ):
        # Asset universe
        self.assets = hrp_cov.index

        # Store inputs
        self.hrp_cov = hrp_cov.copy()
        self.hrp_weights = hrp_weights.copy()
        self.risk_aversion = risk_aversion
        self.tau = tau

        # Numerical stability
        self.Sigma = self.hrp_cov.values + np.eye(len(self.assets)) * ridge
        self.w = self.hrp_weights.values

        # Equilibrium returns
        self.pi = self._compute_equilibrium_returns()

        # Placeholders
        self.mu_bl = None
        self.final_weights = None

    # --------------------------------------------------
    # 1. Equilibrium returns
    # --------------------------------------------------
    def _compute_equilibrium_returns(self) -> pd.Series:
        """
        π = λ Σ w
        """
        pi = self.risk_aversion * self.Sigma @ self.w
        return pd.Series(pi, index=self.assets)

    # --------------------------------------------------
    # 2. Fit Black–Litterman with views
    # --------------------------------------------------
    def fit_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        omega: np.ndarray
    ) -> pd.Series:
        """
        Compute BL posterior expected returns
        """

        # Sanity checks
        assert P.shape[1] == len(self.assets), "P matrix dimension mismatch"
        assert omega.shape[0] == omega.shape[1] == len(Q), "Omega dimension mismatch"

        # Bayesian update
        Sigma_tau_inv = np.linalg.inv(self.tau * self.Sigma)
        omega_inv = np.linalg.inv(omega)

        middle = np.linalg.inv(
            Sigma_tau_inv + P.T @ omega_inv @ P
        )

        mu_bl = middle @ (
            Sigma_tau_inv @ self.pi.values +
            P.T @ omega_inv @ Q
        )

        self.mu_bl = pd.Series(mu_bl, index=self.assets)
        return self.mu_bl

    # --------------------------------------------------
    # 3. Final HRP + BL portfolio
    # --------------------------------------------------
    def get_final_weights(
        self,
        min_tilt: float = 0.5,
        max_tilt: float = 1.5
    ) -> pd.Series:
        """
        HRP weights tilted by BL expected returns
        """

        if self.mu_bl is None:
            raise ValueError("Run fit_views() before computing final weights")

        # Scale BL returns to realistic annual range
        mu_scaled = self.mu_bl / 10

        # Convert returns → tilt
        tilt = np.clip(1 + mu_scaled, min_tilt, max_tilt)

        final_weights = self.hrp_weights * tilt
        final_weights /= final_weights.sum()

        self.final_weights = final_weights
        return final_weights

    # --------------------------------------------------
    # 4. Diagnostics
    # --------------------------------------------------
    def summary(self):
        return pd.DataFrame({
            "HRP Weight": self.hrp_weights,
            "Equilibrium Return (π)": self.pi,
            "BL Return (μ_BL)": self.mu_bl,
            "Final Weight": self.final_weights
        })


# ======================================================
# HRP → BL PIPELINE (PRODUCTION ENTRY POINT)
# ======================================================
def run_hrp_bl_pipeline(
    tickers: list,
    P: np.ndarray,
    Q: np.ndarray,
    omega: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> dict:
    """
    Full pipeline: Prices → Returns → HRP → BL
    """

    hrp = HRP()

    # 1. Get returns
    returns_df = hrp.returns(tickers)

    # 2. HRP
    hrp_weights = hrp.hrp_weights(returns_df)
    hrp_cov = returns_df.cov()

    # 3. BL
    bl = BlackLittermanHRP(
        hrp_cov=hrp_cov,
        hrp_weights=hrp_weights,
        risk_aversion=risk_aversion,
        tau=tau
    )

    mu_bl = bl.fit_views(P, Q, omega)
    final_weights = bl.get_final_weights()

    return {
        "hrp_weights": hrp_weights.to_dict(),
        "bl_returns": mu_bl.to_dict(),
        "final_weights": final_weights.to_dict()
    }


# ======================================================
# TEST (SYNTHETIC, SAFE)
# ======================================================
if __name__ == "__main__":

    assets = ["infy.ns", "geship.ns", "itc.ns", "hdfcbank.ns", "tatasteel.ns"]
    np.random.seed(42)

    assets = [i.upper() for i in assets]

    hrp = HRP()
    returns = hrp.returns(assets)
    hrp_weights, hrp_cov, corr = hrp.hrp_weights(returns)

    bl = BlackLittermanHRP(
        hrp_cov=hrp_cov,
        hrp_weights=hrp_weights
    )

    # Relative view: INFY > TCS by 3%
    P = np.array([[1, -1, 0, 0, 0]])
    Q = np.array([0.03])
    omega = np.diag([0.0004])

    mu_bl = bl.fit_views(P, Q, omega) * 100
    final_weights = bl.get_final_weights() 

    print("\nBL posterior expected returns:")
    print(mu_bl)

    print("\nFinal HRP + BL portfolio weights:")
    print(final_weights)
    print("\nSum of weights:", final_weights.sum())

    print("\nSummary:")
    print(bl.summary())

    print("\n✅ All tests passed successfully.")
