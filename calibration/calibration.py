import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

class Calibration:
    def __init__(self, ticker, start, end, option_type = "C"):
        """
        Initialize the Calibration class with a ticker symbol and date range.

        Parameters:
            ticker : str - Ticker symbol of the asset
            option_type : str - Type of option ("C" for call, "P" for put)
            start : str - Start date for historical data (YYYY-MM-DD)
            end : str - End date for historical data (YYYY-MM-DD)
        """
        self.ticker = ticker
        self.stock = self.get_yf_ticker_data()
        self.option_type = option_type
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

    def get_historical_sigma(self):
        """
        Calibrate the volatility (sigma) from historical data.

        Returns:
            float - Annualized volatility (sigma)
        """
        log_returns = self.compute_log_returns()
        s = self.compute_std_array(log_returns)
        return round(s * np.sqrt(self.T),4) # Annualize the volatility, there are 252 open market days in a year
    
    def get_historical_mu(self):
        """
        Calibrate the drift (mu) from historical data.

        Returns:
            float - Annualized drift (mu)
        """
        sigma = self.get_historical_sigma()
        mu = np.log(self.historical_prices_array[-1] / self.historical_prices_array[0])/self.T + 0.5 * (sigma)**2 # Annualize the drift, there are 252 open market days in a year
        return round(mu,4)
    
    def get_yf_ticker_data(self):
        """
        Fetch and return the Yahoo Finance Ticker object for the specified ticker.

        This method retrieves the Yahoo Finance Ticker object, which provides access to
        various financial data and metadata for the specified ticker symbol.

        Returns:
            yfinance.Ticker: Yahoo Finance Ticker object for the specified ticker.
        """
        return yf.Ticker(self.ticker)

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

    def get_options_dataframe(self):
        """
        Retrieve option data (calls or puts) for the ticker from Yahoo Finance.

        Parameters:
            ticker : str - Ticker symbol of the asset or index (^SPX for the S&P500)

        Returns:
            DataFrame with columns ['tenor', 'strike', 'price'].
        """
        # Fetch the ticker data
        option_dates = self.stock.options  # Available expiration dates

        options_data = []

        # Fetch data for each expiration date
        for option_date in option_dates:
            options_chain = self.stock.option_chain(option_date)
            
            # Select calls or puts
            if self.option_type == 'C':
                option_df = options_chain.calls
            elif self.option_type == 'P':
                option_df = options_chain.puts
            else:
                raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")

            # Add maturity and calculate time to maturity (TTM)
            option_df['maturity'] = option_date
            options_data.append(option_df)

        # Combine all data into a single DataFrame
        all_options = pd.concat(options_data, ignore_index=True)

        # Calculate time to maturity (TTM) in years
        today = datetime.today()
        all_options['tenor'] = all_options['maturity'].apply(
            lambda x: (datetime.strptime(x, "%Y-%m-%d").replace(hour=22, minute=0) - today).total_seconds() / (365 * 24 * 3600)
        )  # 22h = 16h NY time (market close)

        all_options.rename(columns={'lastPrice':'price'}, inplace=True)

        # Filter relevant columns
        all_options = all_options[['tenor', 'strike', 'price']]

        return all_options

    def get_volatility_surface(self):
        """
        Generate a volatility surface using option data.

        Parameters:
            option_type : str - 'call' or 'put' (default: 'call')

        Returns:
            DataFrame with columns ['tenor', 'strike', 'price'].
        """
        # Retrieve option data
        option_data = self.get_options_dataframe()

        # Filter out options with zero or missing prices
        option_data = option_data[option_data['price'] > 0]

        return option_data
    
    def get_treasury_yield_rates(self):
        """
        Fetch US Daily Treasury Par Yield Rates from Yahoo Finance.

        Returns:
            yield_maturities : np.ndarray - Array of yield maturities (in years)
            yields : np.ndarray - Array of corresponding yields (as decimals)
        """
        # Treasury tickers for different maturities
        treasury_tickers = {
            '1M': '^IRX',  # 1-month T-bill
            '3M': '^IRX',  # 3-month T-bill
            '6M': '^IRX',  # 6-month T-bill
            '1Y': '^FVX',  # 1-year Treasury
            '2Y': '^FVX',  # 2-year Treasury
            '5Y': '^FVX',  # 5-year Treasury
            '10Y': '^TNX',  # 10-year Treasury
            '30Y': '^TYX'   # 30-year Treasury
        }

        # Maturities in years
        yield_maturities = np.array([1/12, 3/12, 6/12, 1, 2, 5, 10, 30])

        # Fetch yields for each maturity
        yields = []
        for ticker in treasury_tickers.values():
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                yields.append(data['Close'].iloc[-1] / 100)  # Convert to decimal
            else:
                yields.append(np.nan)  # Handle missing data

        # Remove maturities with missing yields
        valid_indices = ~np.isnan(yields)
        yield_maturities = yield_maturities[valid_indices]
        yields = np.array(yields)[valid_indices]

        return yield_maturities, yields

    def get_treasury_yield_curve(self):
        """
        Generate US Daily Treasury Par Yield Curve Rates

        Returns:
            curve_fit : NelsonSiegelSvenssonCurve using ordinary least squares approach

        """
        yield_maturities, yields = self.get_treasury_yield_rates()
        curve_fit, status = calibrate_nss_ols(yield_maturities,yields)

        return curve_fit
    
    def get_latest_price(self):
        """
        Fetch the latest closing price of the specified ticker.

        Parameters:
            ticker : str - Ticker symbol of the asset

        Returns:
            float - Latest closing price (S0)
        """
        # Fetch the latest data for the specified ticker
        hist = self.stock.history(period="1d")

        # Return the latest closing price
        if not hist.empty:
            return round(hist['Close'].iloc[-1], 4)
        else:
            raise ValueError(f"Unable to fetch the latest price for ticker: {self.ticker}")