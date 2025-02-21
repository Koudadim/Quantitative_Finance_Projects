# -*- coding:utf-8 -*-
#######################################################################
# Here, we implement a class for European Option and Greeks valuation #
# using the Black-Scholes-Merton (BSM) model                          #
# BSMOptionsAndGreeksValuation is the name of the class               #
#######################################################################
import numpy as np
from numpy import log, exp, sqrt
from scipy import stats
from typing import Tuple

class BSMOptionsAndGreeksValuation:
    """
    Valuation of European call options in Black-Scholes-Merton Model (incl. dividend)
    Attributes
    ==========
    St: float
        initial (current) stock/index level
    K: float
        strike price
    t: float
        current time view as the initial time. it is generally equal to 0
    T: float
        time to maturity (in year fractions)
    r: float
        constant risk-free short rate
        assume flat term structure
    sigma: float
        volatility factor in diffusion term
    div_yield: float
        dividend_yield, in percentage %, default = 0.0%
    """

    def __init__(self, St: float, K: float, T: float, r: float, sigma: float, t: float = 0.0, div_yield: float = 0.0):
        assert sigma >= 0, 'volatility cannot be less than zero'
        assert St >= 0, 'initial stock price cannot be less than zero'
        assert 0 <= t <= T, 'the current time t should not be negative nor above the maturity T'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert div_yield >= 0, 'dividend yield cannot be less than zero'

        self.St = float(St)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.t = float(t)
        self.div_yield = float(div_yield)
        
        self._tau = self.T - self.t
        self._D = exp(-self.r * self._tau)
        self._d1, self._d2 = self._calculate_d1_d2()
        self._d3 = None
        self._d4 = None
        self._d5 = None
        self._d6 = None
        self._d7 = None
        self._d8 = None

    def _calculate_d1_d2(self):
        d1 = ((log(self.St / self.K) + (self.r - self.div_yield + 0.5 * self.sigma ** 2) * self._tau) / (
                self.sigma * sqrt(self._tau)))
        d2 = d1 - self.sigma * sqrt(self._tau)

        return d1, d2
    
    def forward_price(self) -> float:
        F = self.St * exp((self.r - self.div_yield) * self._tau)  # Corrected to self.St
        return F
    
    def call_put_values(self) -> Tuple[float, float]:
        """
        :return: call and put options values (tuple)
        """
        call = self._D * (self.forward_price() * stats.norm.cdf(self._d1, 0.0, 1.0) - self.K * stats.norm.cdf(self._d2, 0.0, 1.0))
        put = self._D * (self.K * stats.norm.cdf(-self._d2, 0.0, 1.0) - self.forward_price() * stats.norm.cdf(-self._d1, 0.0, 1.0))
        
        return call, put
    
    def call_put_deltas(self) -> Tuple[float, float]:
        """
        Delta measures the change in the option price for a $1 change in the stock price
        :return: delta of the option
        """
        delta_call = exp(- self.div_yield * self._tau) * stats.norm.cdf(self._d1, 0.0, 1.0)
        delta_put = - exp(- self.div_yield * self._tau) * stats.norm.cdf(-self._d1, 0.0, 1.0)

        return delta_call, delta_put

    def call_put_gamma(self) -> float:
        """
        Gamma measures the change in delta when the stock price changes
        :return: gamma of the option
        """
        gamma = exp(-self.div_yield * self._tau) * stats.norm.pdf(self._d1) / (self.St * self.sigma * sqrt(self._tau))

        return gamma
    
    def call_put_vega(self) -> float:
        """
        Vega measures the change in the option price when volatility changes. Some writers also
        use the terms lambda or kappa to refer to this measure:
        It is common to report vega as the change in the option price per percentage point change
        in the volatility. This requires dividing the vega formula above by 100.
        :return: vega of option
        """
        vega = self.St * exp(-self.div_yield * self._tau) * stats.norm.pdf(self._d1, 0.0, 1.0) * sqrt(self._tau)

        return vega

    def call_put_thetas(self) -> Tuple[float, float]:
        """
        Theta measures the change in the option price with respect to calendar time (t ),
        holding fixed time to expiration (T).

        If time to expiration is measured in years, theta will be the annualized change in the option value.
        To obtain a per-day theta, divide by 252.
        :return: theta of the option
        """
        part1 = self.div_yield * self.St * exp(-self.div_yield * self._tau) * stats.norm.cdf(self._d1)
        part2 = self.r * self.K * stats.norm.cdf(self._d2)
        part3 = (self.St * exp(-self.div_yield * self._tau) * stats.norm.pdf(self._d1) * self.sigma) / (2 * sqrt(self._tau))

        theta_call = part1 - part2 - part3
        theta_put = theta_call + self.r * self.K * self._D - self.div_yield * self.St * exp(-self.div_yield * self._tau)

        return theta_call, theta_put

    

    def call_put_rhos(self) -> Tuple[float, float]:
        """
        Rho is the partial derivative of the option price with respect to the interest rate.
        These expressions for rho assume a change in r of 1.0. We are typically interested in
        evaluating the effect of a change of 0.01 (100 basis points) or 0.0001 (1 basis point). To
        report rho as a change per percentage point in the interest rate, divide this measure by 100.
        To interpret it as a change per basis point, divide by 10,000.
        :return: call_rho, put_rho
        """
        call_rho = self._tau * self.K * self._D * stats.norm.cdf(self._d2)
        put_rho = -self._tau * self.K * self._D * stats.norm.cdf(-self._d2)

        return call_rho, put_rho

    def call_put_psis(self) -> Tuple[float, float]:
        """
        Psi is the partial derivative of the option price with respect to the continuous dividend yield:
        To interpret psi as a price change per percentage point change in the dividend yield, divide
        by 100.
        :return: call_psi, put_psi
        """
        call_psi = - self._tau * self.St * exp(-self.div_yield * self._tau) * stats.norm.cdf(self._d1)
        put_psi = self._tau * self.St * exp(-self.div_yield * self._tau) * stats.norm.cdf(-self._d1)

        return call_psi, put_psi

