import pandas as pd
from datetime import datetime
from calibration.calibration import Calibration
from models.heston import Heston

class Data:
    """
    A class to handle the loading and saving of parameters that are time-consuming to generate live.
    These parameters are saved in CSV files for efficient access.
    """

    def __init__(self, ticker):
        """
        Initialize the Data class with a specific ticker symbol.

        Parameters:
            ticker (str): The ticker symbol for the financial instrument (e.g., '^SPX' for S&P 500).
        """
        self.ticker = ticker

    def save_heston_parameters(self):
        """
        Calibrate the Heston model using historical data and save the calibrated parameters to a CSV file.
        The parameters are saved with a timestamp to track when the calibration was performed.
        """
        # Initialize the Calibration class with the specified ticker and date range
        calibration = Calibration(ticker=self.ticker, start='2022-01-01', end='2023-01-01')

        # Retrieve the Treasury yield curve data
        NSS_curve = calibration.get_treasury_yield_curve()

        # Fetch the options data for the specified ticker and date range
        df = calibration.get_options_dataframe()

        # Calculate the risk-free rates for each option's time to maturity using the Treasury yield curve
        df['rate'] = df['tenor'].apply(NSS_curve)

        # Extract relevant data for calibration
        r = df['rate'].to_numpy('float')  # Risk-free rates
        K = df['strike'].to_numpy('float')  # Strike prices
        tau = df['tenor'].to_numpy('float')  # Times to maturity
        P = df['price'].to_numpy('float')  # Market prices
        S0 = calibration.get_latest_price()  # Latest spot price of the underlying asset

        # Initialize the Heston model with default parameters
        heston = Heston(S0=S0, K=K[0], r=r[0], T=tau[0], option_type="C")

        # Calibrate the Heston model to fit the market prices
        params = heston.calibrate(S0, K, tau, r, P)

        # Generate a timestamp for the calibration date
        calibration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the calibrated parameters and timestamp for saving
        params_dict = {
            "v0": [params[0]],  # Initial variance
            "kappa": [params[1]],  # Mean reversion rate of variance
            "theta": [params[2]],  # Long-term average variance
            "sigma": [params[3]],  # Volatility of volatility
            "rho": [params[4]],  # Correlation between asset price and volatility
            "lambd": [params[5]],  # Risk premium of variance
            "calibration_date": [calibration_date],  # Timestamp of calibration
        }

        # Convert the parameters dictionary to a DataFrame
        params_df = pd.DataFrame(params_dict)

        # Save the DataFrame to a CSV file
        params_df.to_csv("data/heston_calibrated_parameters.csv", index=False)

        print("Calibrated parameters saved to data/heston_calibrated_parameters.csv")
