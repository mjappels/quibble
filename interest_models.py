import numpy as np
from SDE import SDE


class CIR_model(SDE):
    def __init__(self, sigma, theta, alpha, f=1, t0=0, dt=0.001):
        super().__init__(f, t0, dt)
        self.sigma = sigma
        self.theta = theta
        self.alpha = alpha

    def rho(self, t, f):
        return self.sigma(t) * np.sqrt(f)

    def nu(self, t, f):
        return (self.theta(t) - self.alpha(t) * f)


if __name__ == "__main__":
    print("Select model to test:")
    print("1:\tCox-Ingersoll-Ross")
    usrInput = "0"
    while usrInput != "":
        usrInput = input("[1], <RET> to exit: ")
        if usrInput == "1":
            a = CIR_model(lambda x: 1, lambda x: 2, lambda x: 2, 1)
            a.iterate(10).plot((0, 3))
        elif usrInput == "":
            continue
        else:
            print("Invalid input.")
