import pandas as pd
from datetime import datetime
from calibration.calibration import Calibration
from models import Heston, SABR, Dupire

class Data:
    """
    A class to handle the loading and saving of parameters that are time-consuming to generate live.
    These parameters are saved in CSV files for efficient access.
    """

    def __init__(self, ticker, option_type="C"):
        """
        Initialize the Data class with a specific ticker symbol.

        Parameters:
            ticker (str): The ticker symbol for the financial instrument (e.g., '^SPX' for S&P 500).
        """
        self.ticker = ticker
        self.option_type = option_type

    def get_data_for_calibration(self):
        # Initialize the Calibration class with the specified ticker and date range
        calibration = Calibration(ticker=self.ticker, start='2022-01-01', end='2023-01-01', option_type=self.option_type)

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

        return S0, K, r, tau, P
    
    def save_heston_parameters(self):
        """
        Calibrate the Heston model using historical data and save the calibrated parameters to a CSV file.
        The parameters are saved with a timestamp to track when the calibration was performed.
        """
        S0, K, r, tau, P = self.get_data_for_calibration()

        # Initialize the Heston model with default parameters
        heston = Heston(S0=S0, K=K[0], r=r[0], T=tau[0], option_type=self.option_type)

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

    def save_dupire_parameters(self):
        """
        Calibrate the Dupire model using historical data and save the calibrated parameters to a CSV file.
        The parameters are saved with a timestamp to track when the calibration was performed.
        """
        S0, K, r, tau, P = self.get_data_for_calibration()

        # Initialize the Dupire model with default parameters
        dupire = Dupire(S0=S0, K=K[0], r=r[0], T=tau[0], option_type=self.option_type)

        # Calibrate the Dupire model to fit the market prices
        local_vol = dupire.calibrate(K, tau, P)

        # Generate a timestamp for the calibration date
        calibration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the calibrated parameters and timestamp for saving
        params_dict = {
            "local_vol_surface": [local_vol],  # Long-term average variance
            "calibration_date": [calibration_date],  # Timestamp of calibration
        }

        # Convert the parameters dictionary to a DataFrame
        params_df = pd.DataFrame(params_dict)

        # Save the DataFrame to a CSV file
        params_df.to_csv("data/dupire_calibrated_parameters.csv", index=False)

        print("Calibrated parameters saved to data/dupire_calibrated_parameters.csv")

    def save_sabr_parameters(self):
        """
        Calibrate the SABR model using historical data and save the calibrated parameters to a CSV file.
        The parameters are saved with a timestamp to track when the calibration was performed.
        """
        S0, K, r, tau, P = self.get_data_for_calibration()

        # Initialize the SABR model with default parameters
        sabr = SABR(S0=S0, K=K[0], r=r[0], T=tau[0], option_type=self.option_type)
        
        # Calibrate the SABR model to fit the market prices
        params = sabr.calibrate(S0, K, tau, r, P)

        # Generate a timestamp for the calibration date
        calibration_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the calibrated parameters and timestamp for saving
        params_dict = {
            "alpha": [params[0]],  # Volatility level
            "beta": [params[1]],  # Elasticity parameter (0 <= beta <= 1)
            "rho": [params[2]],  # Correlation between asset and volatility.
            "nu": [params[3]],  # Volatility of volatility
            "calibration_date": [calibration_date],  # Timestamp of calibration
        }

        # Convert the parameters dictionary to a DataFrame
        params_df = pd.DataFrame(params_dict)

        # Save the DataFrame to a CSV file
        params_df.to_csv("data/sabr_calibrated_parameters.csv", index=False)

        print("Calibrated parameters saved to data/sabr_calibrated_parameters.csv")

