import numpy as np
from scipy.optimize import minimize
from models.black_scholes import BlackScholes
import warnings
from numpy import ComplexWarning

# Suppress ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)

class SABR:
    """
    Implements the SABR stochastic volatility model for option pricing.
    """

    def __init__(self, S0: float, K: float, r: float, T: float, option_type="C",
                 alpha: float = 0.2, beta: float = 0.5, rho: float = 0.0, nu: float = 0.2):
        """
        Initialize the SABR model parameters.

        :param S0: Initial asset price.
        :param K: Strike price.
        :param r: Risk-free interest rate.
        :param T: Time to maturity.
        :param option_type: Type of option ("C" for call, "P" for put).
        :param alpha: Volatility level.
        :param beta: Elasticity parameter (0 <= beta <= 1).
        :param rho: Correlation between asset and volatility.
        :param nu: Volatility of volatility.
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu

    def implied_vol(self, K):
        """
        Compute the implied volatility for a given strike using the SABR model.

        :param K: Strike price.
        :return: Implied volatility.
        """
        F = self.S0 * np.exp(self.r * self.T)  # Forward price
        alpha = self.alpha
        beta = self.beta
        rho = self.rho
        nu = self.nu

        # ATM case: when F is nearly equal to K.
        if np.isclose(F, K):
            term1 = ((1 - beta) ** 2 / 24) * (alpha ** 2) / (F ** (2 - 2 * beta))
            term2 = (1 / 4) * (rho * beta * nu * alpha) / (F ** (1 - beta))
            term3 = ((2 - 3 * rho ** 2) / 24) * nu ** 2
            vol_atm = alpha / (F ** (1 - beta)) * (1 + (term1 + term2 + term3) * self.T)
            return vol_atm
        else:
            logFK = np.log(F / K)
            FK_avg = (F * K) ** ((1 - beta) / 2)
            z = (nu / alpha) * FK_avg * logFK

            # For small z, revert to the ATM expansion.
            if np.abs(z) < 1e-07:
                return self.implied_vol(F)

            numerator = np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho
            denominator = 1 - rho  # valid since rho < 1
            x_z = np.log(numerator / denominator)

            term1 = alpha / FK_avg * (z / x_z)
            correction = ((1 - beta) ** 2 / 24) * (alpha ** 2) / ((F * K) ** (1 - beta)) \
                         + (rho * beta * nu * alpha) / (4 * (F * K) ** ((1 - beta) / 2)) \
                         + ((2 - 3 * rho ** 2) / 24) * nu ** 2
            term2 = 1 + self.T * correction
            return term1 * term2

    def calculate_option_price(self):
        """
        Calculate the SABR option price.

        Returns:
            float: Option price
        """
        # Retrieve the SABR-implied volatility for the given strike.
        sigma_sabr = self.implied_vol(self.K)
        discount = np.exp(-self.r * self.T)

        if sigma_sabr <= 0:
            if self.option_type.lower() == "call":
                return discount * max(self.S0 - self.K, 0)
            elif self.option_type.lower() == "put":
                return discount * max(self.K - self.S0, 0)
            else:
                raise ValueError("option_type must be either 'call' or 'put'")

        # Use Black-Scholes to price the option with the SABR-implied volatility
        bs = BlackScholes(r=self.r, S=self.S0, K=self.K, T=self.T, sigma=sigma_sabr, option_type=self.option_type)
        return bs.calculate_option_price()
    
    def SqErr(self, params, S0, market_prices, strikes, maturities, r):
        """
        Calculate the sum of squared errors between market prices and SABR model prices.

        Parameters:
            params : np.ndarray - Array of SABR parameters [alpha, beta, rho, nu]
            market_prices : np.ndarray - Array of market option prices
            strikes : np.ndarray - Array of strike prices
            maturities : np.ndarray - Array of maturities
            S0 : float - Initial asset price
            r : np.ndarray - Array of risk-free rates

        Returns:
            float - Sum of squared errors
        """
        alpha, beta, rho, nu = params
        print(f"Params: alpha={alpha}, beta={beta}, rho={rho}, nu={nu}")

        model_prices = []
        for i in range(len(market_prices)):
            sabr_model = SABR(S0=S0,
                                   K=strikes[i],
                                   r=r[i],
                                   T=maturities[i],
                                   option_type="C",  # Assuming call options for calibration
                                   alpha=alpha,
                                   beta=beta,
                                   rho=rho,
                                   nu=nu)
            model_price = sabr_model.calculate_option_price()
            model_prices.append(model_price)

        model_prices = np.array(model_prices)

        # Compute the squared error
        squared_error = np.sum((market_prices - model_prices) ** 2) / len(market_prices)
        print(f"Squared Error: {squared_error}")
        return squared_error

    def calibrate(self, S0, K, tau, r, P):
        """
        Calibrate the SABR model parameters by minimizing the squared error between
        model prices and market prices.

        Parameters:
            S0 (float): Initial asset price.
            K (np.ndarray): Array of strike prices.
            tau (np.ndarray): Array of times to maturity.
            r (np.ndarray): Array of risk-free rates.
            P (np.ndarray): Array of market prices for the options.

        Returns:
            OptimizeResult: Result of the optimization, containing the optimal alpha, beta, rho, and nu parameters.
        """
        # Initial guesses and bounds for the parameters
        initial_params = {
            "alpha": {"x0": self.alpha, "lbub": [1e-6, None]},
            "beta": {"x0": self.beta, "lbub": [0, 1]},
            "rho": {"x0": self.rho, "lbub": [-0.999, 0.999]},
            "nu": {"x0": self.nu, "lbub": [1e-6, None]},
        }

        x0 = [param["x0"] for param in initial_params.values()]
        bnds = [param["lbub"] for param in initial_params.values()]

        # Minimize the squared error function
        result = minimize(
            self.SqErr,
            x0,
            args=(S0, P, K, tau, r),
            method='L-BFGS-B',
            bounds=bnds,
            tol=1e-6,
            options={'maxiter': int(1e6)}
        )

        return [param for param in result.x]