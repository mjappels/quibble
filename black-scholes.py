from brownian import Brownian
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class BlackScholes:
    def __init__(self, volatility, drift, r, T, s, n=1000):
        self.W = Brownian(T, n).path
        self.v = volatility
        self.m = drift
        self.r = r
        self.T = T
        self.s = s
        self.dt = 1 / n
        self.tsteps = np.arange(T * n) * self.dt
        self.stock = self.s * np.exp(self.v * self.W + self.m * self.tsteps)
        self.bond = np.exp(r * self.tsteps)

    def callValue(self, k, t):
        p1 = self.stock * \
            norm.cdf((np.log(self.stock / k) + (self.r + self.v**2 / 2)
                      * (self.T - t)) / (self.v * np.sqrt(self.T - t)))
        p2 = -k * np.exp(-self.r * (self.T - t)) * norm.cdf((np.log(self.stock / k) +
                                                             (self.r - self.v**2 / 2) * (self.T-t)) / (self.v * np.sqrt(self.T - t)))
        return p1 + p2

    def call(self, k):
        return self.callValue(k, self.tsteps)

    def hedge(self, k):
        stockHedge = norm.cdf((np.log(self.stock / k) + (self.r + self.v**2 / 2) * (
            self.T - self.tsteps)) / (self.v * np.sqrt(self.T - self.tsteps)))
        bondHedge = -k * np.exp(-self.r * self.T) * norm.cdf((np.log(self.stock / k) + (
            self.r - self.v**2 / 2) * (self.T - self.tsteps)) / (self.v * np.sqrt(self.T - self.tsteps)))
        return np.array([stockHedge, bondHedge])


if __name__ == "__main__":
    a = BlackScholes(0.178, 0.087, 0.05, 30, 100, 100)
    plt.plot(a.tsteps, a.stock, 'b-')
    plt.plot(a.tsteps, a.call(2500), 'r-')
    plt.plot(a.tsteps, a.hedge(2500)[0], 'g-')
    plt.show()
