import datetime

import numpy


class Bond:
    def __init__(self, cash_flow_dates, cash_flows):
        self._cash_flow_dates = cash_flow_dates
        self._cash_flows = cash_flows

    def pv(self, date: datetime.date, discount_curve) -> float:
        years_to_flow = numpy.fromiter(
            ((flow_date - date).days for flow_date in self._cash_flow_dates),
            count=len(self._cash_flow_dates),
            dtype='int'
        )

        return numpy.sum(self._cash_flows * discount_curve(years_to_flow), where=years_to_flow > 0)
