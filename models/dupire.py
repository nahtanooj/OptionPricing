import numpy as np
from scipy.interpolate import griddata

class Dupire:
    """
    Implements the Dupire local volatility model for option pricing.
    """

    def __init__(self, S0, K, r, T, option_type="C"):
        """
        Initialize the Dupire model parameters.

        :param S0: Initial asset price.
        :param K: Strike price.
        :param r: Risk-free interest rate.
        :param T: Time to maturity.
        :param option_type: Type of option ("C" for call, "P" for put).
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.option_type = option_type

    def local_volatility(self, strikes, maturities, prices, S0, r):
        """
        Calculate the local volatility surface using Dupire's formula.

        :param strikes: Array of strike prices.
        :param maturities: Array of maturities.
        :param prices: Array of market prices for call options.
        :param S0: Initial asset price.
        :param r: Risk-free interest rate.

        :return: Local volatility surface.
        """
        # Create a grid for interpolation
        points = np.array((strikes, maturities)).T
        values = prices

        # Interpolate call prices to create a smooth surface
        grid_K, grid_T = np.meshgrid(np.linspace(min(strikes), max(strikes), 100),
                                     np.linspace(min(maturities), max(maturities), 100))
        grid_call_prices = griddata(points, values, (grid_K, grid_T), method='cubic')

        # Compute partial derivatives using finite differences
        dC_dT = np.gradient(grid_call_prices, axis=1)
        dC_dK = np.gradient(grid_call_prices, axis=0)
        d2C_dK2 = np.gradient(dC_dK, axis=0)

        # Apply Dupire's formula
        local_vol_surface = np.sqrt(2 * (dC_dT + r * grid_K * dC_dK) / (grid_K**2 * d2C_dK2))

        return grid_K, grid_T, local_vol_surface

    def calibrate(self, strikes, maturities, market_prices):
        """
        Calibrate the Dupire model to market prices.

        :param market_prices: Array of market prices for options.
        :param strikes: Array of strike prices.
        :param maturities: Array of maturities.

        :return: Calibrated local volatility surface.
        """
        # Calibrate the local volatility surface using market prices
        grid_K, grid_T, local_vol_surface = self.local_volatility(strikes, maturities, market_prices, self.S0, self.r)

        return grid_K, grid_T, local_vol_surface

    def calculate_option_price(self, local_vol_surface):
        """
        Calculate the option price using the local volatility surface.

        :param K: Strike price.
        :param T: Time to maturity.
        :param local_vol_surface: Calibrated local volatility surface.

        :return: Option price.
        """
        # Interpolate the local volatility surface to get the local volatility for the given K and T
        local_vol = griddata(
            (self.grid_K.flatten(), self.grid_T.flatten()),
            local_vol_surface.flatten(),
            (self.K, self.T),
            method='cubic'
        )

        # Use the local volatility to price the option (e.g., using Monte Carlo simulation)
        # For simplicity, we'll use a placeholder formula here
        price = self.S0 * np.exp(-self.r * self.T) * local_vol  # Placeholder pricing formula

        return price
