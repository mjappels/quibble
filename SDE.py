import numpy as np
import matplotlib.pyplot as plt


class SDE:
    def __init__(self, f=0, t0=0, dt=0.001):
        """ Initialise and set the parameters for an ODE integration.

        :param f  : initial population.
        :param t0 : initial time.
        :param dt : integration time step.
        :param sub_dt : number of subdivisions in dt for brownian.
        """
        self.reset(f, t0, dt)

    def reset(self, f, t0, dt):
        """ Reset the integration parameters; see __init__ for more info."""
        self.f = f
        self.t = t0
        self.dt = dt
        self.t_list = []   # to store N values for plots
        self.f_list = []   # to store t values for plots

    def rho(self, t, f):
        """ volatility of sde """
        pass

    def nu(self, t, f):
        """ drift of sde """
        pass

    def one_step(self):
        """ Perform a single integration step using the Euler method."""
        self.dW = np.random.normal(0, np.sqrt(self.dt))
        self.f += self.rho(self.t, self.f) * self.dW + self.nu(self.t, self.f) * self.dt
        self.t += self.dt

    def iterate(self, tmax):
        """ Solve the equation df(t) = rho()dW + nu()dt  until time tmax.
            Update f, t and append all values to N_list and t_list.

        :param tmax : upper bound of integration.
        """

        while(self.t < tmax):
            self.one_step()
            self.f_list.append(self.f)
            self.t_list.append(self.t)

        return self

    def plot(self, plotRange="", style='b-'):
        """ Display function N(t).

        :param style: matplotlib style string for the plot.
        """
        plt.plot(self.t_list, self.f_list, style)
        if plotRange:
            plt.ylim(*plotRange)
        plt.show()


if __name__ == "__main__":
    class test_SDE(SDE):
        def __init__(self, rho, nu, f=1, t0=0, dt=0.001):
            super().__init__(f, t0, dt)
            self.rhov = rho
            self.nuv = nu

        def rho(self, t, f):
            return self.rhov

        def nu(self, t, f):
            return self.nuv

    x = test_SDE(1, 0, 5)
    x.iterate(10).plot((0, 15))
