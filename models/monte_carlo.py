import numpy as np
from scipy.stats import norm

class MonteCarlo:
    def __init__(self, S_0, K, T, mu, sigma, option_type='C', confidence_level=0.99, n_simul=10000):
        """
        Initialize the Monte Carlo simulation parameters.

        Parameters:
        S_0 : float - Initial price of the underlying asset
        K : float - Strike price
        T : float - Time to maturity (in years)
        mu : float - Drift (expected return)
        sigma : float - Volatility of the underlying asset
        option_type : str - Type of option ("C" for call, "P" for put)
        confidence_level : float - Confidence level for the confidence interval (default: 0.99)
        n_simul : int - Number of simulations (default: 10,000)
        """
        self.S_0 = S_0
        self.K = K
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.option_type = option_type
        self.confidence_level = confidence_level
        self.n_simul = n_simul
        self.payoffs = []
        self.mu_k = []
        self.sigma_k = []
        self.lowerbound = []
        self.upperbound = []

    def generate_alea(self, n): 
        """
        Generate a gaussian vector of size n.

        Parameters:
        n: The size of the vector to be returned

        Return:
        A Gaussian array
        """
        return np.random.normal(size=n)
    
    def generate_wiener_process(self):
        """
        Generate a Wiener process (Gaussian noise) for the simulation.
        """
        return np.random.normal()

    def compute_underlying_final_price(self, w_t):
        """
        Compute the final price of the underlying asset at time T.

        Parameters:
        w_t : float - Wiener process value

        Returns:
        float - Simulated price of the underlying asset at time T
        """
        return self.S_0 * np.exp((self.mu - (self.sigma**2) / 2) * self.T + self.sigma * w_t)

    def compute_payoff_call(self, S_T):
        """
        Compute the payoff of the call option.

        Parameters:
        S_T : float - Simulated price of the underlying asset at time T

        Returns:
        float - Payoff of the call option
        """
        return max(S_T - self.K, 0)

    def compute_payoff_put(self, S_T):
        """
        Compute the payoff of the put option.

        Parameters:
        S_T : float - Simulated price of the underlying asset at time T

        Returns:
        float - Payoff of the put option
        """
        return max(self.K - S_T, 0)
    
    def compute_payoff(self, S_T):
        """
        Compute the payoff of the option.

        Parameters:
        S_T : float - Simulated price of the underlying asset at time T

        Returns:
        float - Payoff of the option
        """
        if self.option_type == "C":
            return self.compute_payoff_call(S_T)
        elif self.option_type == "P":
            return self.compute_payoff_put(S_T)
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")

    def compute_confidence_interval(self):
        """
        Compute the z-value corresponding to the confidence level.

        Returns:
        float - Z-value for the confidence interval
        """
        tail_probability = (1 - self.confidence_level) / 2
        return norm.ppf(1 - tail_probability)

    def compute_delta(self):
        """
        Compute the difference between the last lower bound and upper bound as a percentage of the mean option price.

        Returns:
        float - The width of the confidence interval as a percentage of the mean option price.
        """
        if len(self.lowerbound) == 0 or len(self.upperbound) == 0:
            raise ValueError("Simulation has not been run yet. Call run_simulation() first.")
        
        # Get the mean option price (mu_k)
        mean_option_price = self.mu_k[-1]
        
        # Avoid division by zero
        if mean_option_price == 0:
            raise ValueError("Mean option price is zero. Cannot compute percentage difference.")
        
        # Compute the percentage difference
        delta = ((self.upperbound[-1] - self.lowerbound[-1]) / mean_option_price) * 100
        
        return delta

    def run_simulation(self):
        """
        Run the Monte Carlo simulation and store results.
        """
        a = self.compute_confidence_interval()

        for i in range(self.n_simul):
            w_t = self.generate_wiener_process()
            S_T = self.compute_underlying_final_price(w_t)
            payoff = self.compute_payoff(S_T)
            self.payoffs.append(payoff)

            # Update sliding mean
            self.mu_k.append(np.mean(self.payoffs))

            # Update sliding standard deviation only if there are at least 2 data points
            if i >= 1:
                self.sigma_k.append(np.std(self.payoffs, ddof=1))
            else:
                self.sigma_k.append(0)  # Set standard deviation to 0 for the first iteration

            # Update confidence interval bounds
            if i >= 1:
                self.lowerbound.append(self.mu_k[-1] - a * (self.sigma_k[-1] / np.sqrt(i + 1)))
                self.upperbound.append(self.mu_k[-1] + a * (self.sigma_k[-1] / np.sqrt(i + 1)))
            else:
                self.lowerbound.append(self.mu_k[-1])  # No confidence interval for the first iteration
                self.upperbound.append(self.mu_k[-1])

    def get_option_price(self):
        """
        Run n simulations and get the Monte Carlo estimate of the option price.

        Returns:
        float - Estimated option price
        """
        self.run_simulation()
        return np.mean(self.payoffs)