import numpy as np
import yfinance as yf

class Calibration:
    def __init__(self, ticker, start, end):
        """
        Initialize the Calibration class with a ticker symbol and date range.

        Parameters:
            ticker : str - Ticker symbol of the asset
            start : str - Start date for historical data (YYYY-MM-DD)
            end : str - End date for historical data (YYYY-MM-DD)
        """
        self.ticker = ticker
        self.start_date = start
        self.end_date = end
        self.T = 252  # Assuming 252 trading days in a year
        self.historical_prices_df = self.get_historical_prices_df()
        self.historical_prices_array = self.get_historical_prices_array()

    def get_historical_prices_df(self):
        """
        Fetch historical prices from Yahoo Finance.

        Returns:
            Pandas Dataframe: Historical closing prices
        """
        df = yf.download(self.ticker, self.start_date, self.end_date, progress=False)
        return df[['Close']]
    
    def get_historical_prices_array(self):
        """
        Convert the historical prices DataFrame from Yahoo Finance into an ndarray.

        Returns:
            numpy.ndarray: Historical closing prices
        """
        return self.historical_prices_df.values.flatten()

    def compute_log_returns(self):
        """
        Calculate the logarithmic returns of a given time series of asset prices.

        Returns:
            numpy.ndarray: Logarithmic returns.
        """
        log_returns = np.log(self.historical_prices_array[1:] / self.historical_prices_array[:-1])
        return log_returns
    
    def compute_mean_array(self, array):
        """
        Calculate the mean of an array
        
        Returns:
            Float: The mean of the array
        """
        return np.mean(array)
    
    def compute_std_array(self, array):
        """
        Calculate the standard deviation of an array
        
        Returns:
            Float: The standard deviation of the array
        """
        return np.std(array, ddof=1) # ddof=1 for sample standard deviation

    def calibrate_historical_sigma(self):
        """
        Calibrate the volatility (sigma) from historical data.

        Returns:
            float - Annualized volatility (sigma)
        """
        log_returns = self.compute_log_returns()
        s = self.compute_std_array(log_returns)
        return round(s * np.sqrt(self.T),4) # Annualize the volatility, there are 252 open market days in a year
    
    def calibrate_historical_mu(self):
        """
        Calibrate the drift (mu) from historical data.

        Returns:
            float - Annualized drift (mu)
        """
        sigma = self.calibrate_historical_sigma()
        mu = np.log(self.historical_prices_array[-1] / self.historical_prices_array[0])/self.T + 0.5 * (sigma)**2 # Annualize the drift, there are 252 open market days in a year
        return round(mu,4)

    def get_risk_free_rate(self):
        """
        Fetch the risk-free rate using the 13-week Treasury Bill (^IRX) or the 10-year Treasury Bond (^TNX) as a fallback.

        Returns:
            float - Risk-free rate (annualized)
        """
        # Fetch 13-week Treasury Bill data (^IRX)
        tbill = yf.Ticker("^IRX")
        tbill_data = tbill.history(period="1d")

        # If 13-week T-Bill data is unavailable, use 10-year Treasury Bond (^TNX) as a fallback
        if tbill_data.empty:
            tbond = yf.Ticker("^TNX")
            tbond_data = tbond.history(period="1d")
            if tbond_data.empty:
                raise ValueError("Unable to fetch risk-free rate data.")
            risk_free_rate = tbond_data['Close'].iloc[-1] / 100
        else:
            risk_free_rate = tbill_data['Close'].iloc[-1] / 100

        return round(risk_free_rate, 4)

    def get_latest_price(self):
        """
        Fetch the latest closing price of the stock.

        Returns:
            float - Latest closing price (S_0)
        """
        # Fetch the latest closing price from the historical data
        return round(self.historical_prices_df['Close'].iloc[-1].iloc[0],4)