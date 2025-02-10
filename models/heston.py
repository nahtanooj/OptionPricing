import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import warnings
from numpy import ComplexWarning

# Suppress ComplexWarning
warnings.filterwarnings("ignore", category=ComplexWarning)

class Heston:
    """
    Implements the Heston stochastic volatility model for option pricing.
    """

    def __init__(self, S0: float, K: float, r: float, T: float, option_type="C", 
                 v0: float = 0.04, kappa: float = 2.0, theta: float = 0.04, 
                 sigma: float = 0.5, rho: float = -0.7, lambd: float = 0.0):        
        """
        Initialize the Heston model parameters.

        :param S0: Initial asset price.
        :param K: Strike price.
        :param v0: Initial variance.
        :param kappa: Mean reversion rate of variance.
        :param theta: Long-term average variance.
        :param sigma: Volatility of volatility.
        :param lambd: risk premium of variance.
        :param rho: Correlation between asset price and volatility.
        :param r: Risk-free interest rate.
        :param T: Time to maturity.
        """
        self.S0 = S0
        self.K = K
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.lambd = lambd
        self.rho = rho
        self.r = r
        self.T = T
        self.option_type = option_type

    
    def compute_rspi(self, phi):
        """
        Returns rho*sigma*phi*i which is a number that we will be using a lot

        """
        return self.rho*self.sigma*phi*1j
    
    def compute_constant_a(self):
        """
        Returns the a constant in the Heston model

        """
        return self.kappa * self.theta

    def compute_constant_b(self):
        """
        Returns the b constant in the Heston model

        """
        return self.kappa + self.lambd
    
    def compute_constants(self, phi):
        """
        Returns the a, b and rpsi constants we will need to compute the characteric function

        """
        a = self.compute_constant_a()
        b = self.compute_constant_b()
        rspi = self.compute_rspi(phi)
        return (a,b,rspi)
    
    def compute_d(self, phi):
        """
        Define the d parameter given phi and b

        """
        rspi = self.compute_rspi(phi)
        b = self.compute_constant_b()
        return np.sqrt( (rspi-b)**2 + (phi*1j+phi**2)*self.sigma**2)
    
    def compute_g(self, rspi, b, d):
        """
        Define the g parameter given phi, b and d

        """
        return (b-rspi+d)/(b-rspi-d)
    
    def compute_parameters(self, phi, b, rspi):
        """
        Returns the d and g parameters we will need to compute the characteric function

        """
        d = self.compute_d(phi)
        g = self.compute_g(rspi, b, d)
        return (d,g)
    
    def compute_first_exponential_in_cf(self, phi):
        """
        Define the first exponential in the characteristic function

        """
        return np.exp(self.r*phi*1j*self.T) 
    
    def compute_second_exponential_in_cf(self, a, b, rspi, d, g):
        """
        Define the second exponential in the characteristic function

        """
        return np.exp(a*self.T*(b-rspi+d)/self.sigma**2 + self.v0*(b-rspi+d)*( (1-np.exp(d*self.T))/(1-g*np.exp(d*self.T)) )/self.sigma**2)
    
    def compute_second_term_in_cf(self, phi, a, d, g):
        """
        Define the second term (non exponential) in the characteristic function

        """
        return self.S0**(phi*1j)*( (1-g*np.exp(d*self.T))/(1-g) )**(-2*a/self.sigma**2)
    
    def compute_characteristic_function(self, phi):
        """
        Compute the characteristic function by components

        """
        a, b, rspi = self.compute_constants(phi)
        d, g = self.compute_parameters(phi, b, rspi)

        # components of the characteristic function
        exp1 = self.compute_first_exponential_in_cf(phi)
        term2 = self.compute_second_term_in_cf(phi, a, b, g)
        exp2 = self.compute_second_exponential_in_cf(a, b, rspi, d, g)

        return exp1*term2*exp2
    
    def integrand(self, phi):
        numerator = np.exp(self.r*self.T)*self.compute_characteristic_function(phi-1j) - self.K*self.compute_characteristic_function(phi)
        denominator = 1j*phi*self.K**(1j*phi)
        return numerator/denominator
    
    def calculate_option_price(self):
        """
        Calculate the Heston option price.

        Returns:
            float: Option price
        """

        real_integral, err = np.real( quad(self.integrand, 0, 100) )

        if self.option_type == "C":
            # Call option price
            price = (self.S0 - self.K*np.exp(-self.r*self.T))/2 + real_integral/np.pi
        elif self.option_type == "P":
            # Put option price
            price = (self.K*np.exp(-self.r*self.T)- self.S0)/2 + real_integral/np.pi
        else:
            raise ValueError("Invalid option type. Use 'C' for Call or 'P' for Put.")
        return price

    def SqErr(self, params, S0, market_prices, strikes, maturities, r):
        """
        Calculate the sum of squared errors between market prices and Heston model prices.

        Parameters:
            params : np.ndarray - Array of Heston parameters [v0, kappa, theta, sigma, rho, lambd]
            market_prices : np.ndarray - Array of market option prices
            strikes : np.ndarray - Array of strike prices
            maturities : np.ndarray - Array of maturities
            S0 : float - Initial asset price
            r : np.ndarray - Array of risk-free rates

        Returns:
            float - Sum of squared errors
        """
        v0, kappa, theta, sigma, rho, lambd = params
        print(f"Params: v0={v0}, kappa={kappa}, theta={theta}, sigma={sigma}, rho={rho}, lambd={lambd}")
        
        model_prices = []
        for i in range(len(market_prices)):
            heston_model = Heston(S0=S0,
                                  K=strikes[i],
                                  v0=v0,
                                  kappa=kappa,
                                  theta=theta,
                                  sigma=sigma,
                                  lambd=lambd,
                                  rho=rho,
                                  r=r[i],
                                  T=maturities[i])
            model_price = heston_model.calculate_option_price()
            model_prices.append(model_price)

        model_prices = np.array(model_prices)

        # Compute the squared error
        squared_error = np.sum((market_prices - model_prices) ** 2) / len(market_prices)
        print(f"Squared Error: {squared_error}")
        return squared_error

    def calibrate(self, S0, K, tau, r, P):
        """
        Calibrate the Heston model parameters by minimizing the squared error between
        model prices and market prices.

        Parameters:
            S0 (float): Initial asset price.
            K (np.ndarray): Array of strike prices.
            tau (np.ndarray): Array of times to maturity.
            r (np.ndarray): Array of risk-free rates.
            P (np.ndarray): Array of market prices for the options.

        Returns:
            OptimizeResult: Result of the optimization, containing the optimal v0, kappa, theta, sigma and lambd parameters.
        """
        # Initial guesses and bounds for the parameters
        initial_params = {
            "v0": {"x0": self.v0, "lbub": [1e-3, 0.5]},
            "kappa": {"x0": self.kappa, "lbub": [0.1, 10]},
            "theta": {"x0": self.theta, "lbub": [1e-3, 0.5]},
            "sigma": {"x0": self.sigma, "lbub": [0.1, 2]},
            "rho": {"x0": self.rho, "lbub": [-1, 1]},
            "lambd": {"x0": self.lambd, "lbub": [-1, 1]},
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