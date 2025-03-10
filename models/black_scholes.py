import numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Implements the Black-Scholes-Merton model for option pricing.
    """
        
    def __init__(self, r, S, K, T, sigma, option_type="C"):
        """
        Initialize the BlackScholes model parameters.

        Parameters:
            r : float - Risk-free interest rate (annualized)
            S : float - Current stock price
            K : float - Strike price
            T : float - Time to maturity (in years)
            sigma : float - Volatility of the underlying asset (annualized)
            option_type : str - Type of option ("C" for call, "P" for put)
        """
        self.r = r
        self.S = S
        self.K = K
        self.T = T
        self.sigma = sigma
        self.option_type = option_type

    def calculate_d1_d2(self):
        """
        Calculate d1 and d2 for the Black-Scholes formula.

        Returns:
            tuple: (d1, d2)
        """
        d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def calculate_option_price(self):
        """
        Calculate the Black-Scholes option price.

        Returns:
            float: Option price
        """
        d1, d2 = self.calculate_d1_d2()

        if self.option_type == "C":
            # Call option price
            price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == "P":
            # Put option price
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")

        return price
    
    def calculate_vega(self):
        """
        Calcule le vega de l'option (sensibilité du prix à la volatilité).

        Returns:
            float: Vega
        """
        d1, _ = self.calculate_d1_d2()
        return self.S * norm.pdf(d1) * np.sqrt(self.T)