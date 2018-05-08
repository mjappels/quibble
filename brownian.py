import numpy as np
import matplotlib.pyplot as plt


class RandomWalk:
    def __init__(self, n, step=1, p=0.5):
        self.n = n
        self.step = step
        self.p = p
        self.genPath(self.n, self.step, self.p)

    def genPath(self, n, step=1, p=0.5):
        '''
        generates an n-step random walk with up-probablility p.
        '''
        self.path = np.cumsum(
            np.where(np.random.random(n) < p, step, -step))

    def plot(self, style='b-'):
        plt.plot(np.arange(self.n), self.path, style)
        plt.show()


class Brownian(RandomWalk):
    def __init__(self, T=1, n=1000):
        super().__init__(T * n, 1 / np.sqrt(n), 0.5)
        self.T = T
        self.n = n
        self.dt = T / n

    def plot(self, style='b-'):
        plt.plot(np.arange(self.T * self.n) * self.dt, self.path, style)
        plt.show()


if __name__ == '__main__':
    a = RandomWalk(50)
    a.plot()
    b = Brownian(3, 1000)
    b.plot()
