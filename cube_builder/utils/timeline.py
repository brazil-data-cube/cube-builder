#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2020 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define Data Cube Timeline utilities."""

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Union


SchemaType = Union['cyclic', 'continuous']


class Intervals:
    """Define a simple abstraction for Data Cube Timeline Intervals.

    A interval can be represented as following::
        intervals = ['mm-dd_mm-dd', 'mm-dd_mm-dd']
    """

    def __init__(self, intervals):
        """Build a timeline interval."""
        self.intervals = intervals

    def get_indice(self, ref_date) -> int:
        """Try to get an indice for interval."""
        indice_interval = 0
        for i in self.intervals:
            if f'{ref_date.month:02d}-{ref_date.day:02d}' == i.split('_')[0]:
                indice_interval = self.intervals.index(i)
                break

        return indice_interval if indice_interval < len(self.intervals) else 0

    def get_element(self, indice) -> dict:
        """Get an time line interval."""
        interval = self.intervals[indice if indice< len(self.intervals) else 0]
        return dict(
            start=interval.split('_')[0],
            end=interval.split('_')[1]
        )

    def get_date(self, ref_date, element, sum_year=True) -> date:
        """Get a date object from given timeline reference."""
        interval_month = int(element.split('-')[0])
        interval_day = int(element.split('-')[1])

        if sum_year and ref_date.month > interval_month:
            return date(ref_date.year + 1, interval_month, interval_day)

        elif not sum_year and ref_date.month < interval_month:
            return date(ref_date.year - 1, interval_month, interval_day)

        else:
            return date(ref_date.year, interval_month, interval_day)


class Timeline:
    """Define a simple abstraction to generate Data Cube timelines."""

    def __init__(self, schema: SchemaType, start_date, end_date, unit, step, cycle=None, intervals=None):
        """Build the timeline object."""
        self.schema = schema
        self.start_date = start_date
        self.end_date = end_date
        self.unit = unit
        self.step = int(step)
        self.cycle = cycle
        self.intervals = intervals

    def _get_first_day_period(self, ref_date, unit=None, intervals=None):
        if unit:
            month = 1 if unit == 'year' else ref_date.month
            return ref_date.replace(day=1, month=month)

        elif intervals:
            start_element = intervals.get_element(0)['start']
            return intervals.get_date(ref_date, start_element, sum_year=False)

        else:
            return ref_date

    def _get_last_day_period(self, ref_date, step, unit, intervals=None):
        if not intervals:
            return self._next_step(ref_date, step, unit) - timedelta(days=1)

        else:
            indice = intervals.get_indice(ref_date)
            end_element = intervals.get_element(indice)['end']
            return intervals.get_date(ref_date, end_element)

    def _next_step(self, last_date, step=None, unit=None, intervals=None):
        if intervals:
            indice = intervals.get_indice(last_date)
            start_element = intervals.get_element(indice + 1)['start']
            return intervals.get_date(last_date, start_element)

        else:
            if unit == 'day':
                period = last_date + relativedelta(days=step)
                return date(period.year, period.month, period.day)

            elif unit == 'month':
                period = last_date + relativedelta(months=step)
                return date(period.year, period.month, 1)

            elif unit == 'year':
                period = last_date + relativedelta(years=step)
                return date(period.year, 1, 1)

    def _decode_period_continuous(self, start_date, end_date, unit, step, cut_start=None, cut_end=None,
                                  intervals=None, full_period=True, relative=False):
        start_period = start_date
        end_period = self._get_last_day_period(start_period, step, unit, intervals)

        # mount all periods
        if relative and intervals:
            periods = []
        else:
            periods = [[start_period, end_period]]

        while True:
            start_period = self._next_step(start_period, step, unit, intervals)
            end_period = self._get_last_day_period(start_period, step, unit, intervals)

            if start_date <= start_period and end_date >= end_period:
                periods.append([start_period, end_period])

            elif start_date <= start_period and end_date < end_period and start_period <= end_date:
                if not intervals and not full_period:
                    periods.append([start_period, end_date])
                else:
                    periods.append([start_period, end_period])

            if end_period > end_date:
                break

            if relative and intervals:
                stop = False
                for _, _end_period in periods:
                    if _end_period == end_period:
                        stop = True
                        break

                if stop:
                    break

        # cut periods
        result = []
        if cut_start and cut_end:
            for period in periods:
                if period[0] >= cut_start and period[0] <= cut_end:
                    result.append(period)
        else:
            result = periods

        # set `cut range` if not results
        if not result:
            result.append([cut_start, cut_end])
            
        return result

    def _decode_period_cyclic(self, start_date, end_date, unit, step, cyclic_unit, cyclic_step, cyclic_interval=None):
        periods = []

        periods_cyclic = self._decode_period_continuous(self._get_first_day_period(start_date, unit=cyclic_unit), 
                                                        end_date, cyclic_unit, cyclic_step, intervals=cyclic_interval,
                                                        relative=True)

        for period_cyclic in periods_cyclic:
            if cyclic_interval:
                for interval in cyclic_interval.intervals:
                    cut_start = datetime.strptime(f'{period_cyclic[0].year}-{interval.split("_")[0]}', '%Y-%m-%d').date()
                    cut_end = datetime.strptime(f'{period_cyclic[1].year}-{interval.split("_")[1]}', '%Y-%m-%d').date()
                    periods += self._decode_period_continuous(start_date, end_date, unit, step, cut_start, cut_end,
                                                              intervals=cyclic_interval, relative=True)
            else:
                periods += self._decode_period_continuous(period_cyclic[0], period_cyclic[1], unit, step, start_date, end_date, full_period=False)

        return periods

    def mount(self):
        """Mount a time line using the Timeline constructor parameters."""
        if self.schema.lower() == 'cyclic':
            intervals = Intervals(self.cycle['intervals']) if self.cycle.get('intervals') else None
            periods = self._decode_period_cyclic(self.start_date, self.end_date, self.unit, self.step, 
                                                 self.cycle['unit'], int(self.cycle['step']), intervals)
        else:
            intervals = Intervals(self.intervals) if self.intervals else None
            start_date = self._get_first_day_period(self.start_date, intervals=intervals)

            cut_start = start_date if not self.intervals else None
            cut_end = self.end_date if not self.intervals else None
            periods = self._decode_period_continuous(start_date, self.end_date, self.unit, self.step,
                                                     cut_start, cut_end, intervals)

        return periods
