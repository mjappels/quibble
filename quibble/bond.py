import datetime

import numpy
import scipy.optimize


class Bond:
    def __init__(self, cash_flow_dates, cash_flows):
        self._cash_flow_dates = cash_flow_dates
        self._cash_flows = cash_flows

    def pv(self, date: datetime.date, discount_curve) -> float:
        years_to_flow = numpy.fromiter(
            ((flow_date - date).days / 365 for flow_date in self._cash_flow_dates),
            count=len(self._cash_flow_dates),
            dtype='float'
        )

        return numpy.sum(self._cash_flows * discount_curve(years_to_flow), where=years_to_flow > 0)

    def get_yield(self, date: datetime.date, pv: float) -> float:
        years_to_flow = numpy.fromiter(
            ((flow_date - date).days / 365 for flow_date in self._cash_flow_dates),
            count=len(self._cash_flow_dates),
            dtype='float'
        )

        def f(y):
            return numpy.sum(self._cash_flows * numpy.exp(-y * years_to_flow), where=years_to_flow > 0) - pv

        return scipy.optimize.fsolve(f, 0)[0]
