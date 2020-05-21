import numpy


class DiscountCurveFactory:
    def __init__(self, interpolant):
        self._interpolant = interpolant

    def invoke(self, x, rates):
        rates_interp = self._interpolant(x, rates)

        def spot(n_years):
            return numpy.exp(-rates_interp(n_years) * n_years)

        return spot
