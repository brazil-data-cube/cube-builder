#
# This file is part of Cube Builder.
# Copyright (C) 2022 INPE.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/gpl-3.0.html>.
#

"""Define Data Cube Timeline utilities."""

from datetime import date, datetime, timedelta
from typing import List, Union

from dateutil.relativedelta import relativedelta

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

    def get_date(self, ref_date, element, sum_year=True, next=False) -> date:
        """Get a date object from given timeline reference."""
        interval_month = int(element.split('-')[0])
        interval_day = int(element.split('-')[1])

        if sum_year and ref_date.month > interval_month:
            return date(ref_date.year + 1, interval_month, interval_day)

        elif not sum_year and ref_date.month < interval_month:
            return date(ref_date.year - 1, interval_month, interval_day)

        else:
            if next:
                return date(ref_date.year + 1, interval_month, interval_day)

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

    def _get_last_day_period(self, ref_date, step, unit, intervals=None, next=False):
        if not intervals:
            return self._next_step(ref_date, step, unit) - timedelta(days=1)

        else:
            indice = intervals.get_indice(ref_date)
            end_element = intervals.get_element(indice)['end']
            return intervals.get_date(ref_date, end_element, next=next)

    def _next_step(self, last_date, step=None, unit=None, intervals=None, next=False):
        if intervals:
            indice = intervals.get_indice(last_date)
            start_element = intervals.get_element(indice + 1)['start']
            return intervals.get_date(last_date, start_element, next=next)

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
        periods = [[start_period, end_period]]

        while True:
            start_period = self._next_step(start_period, step, unit, intervals, next=relative)
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
        if self.start_date > self.end_date:
            raise ValueError(f'The End date "{self.end_date}" must not be lower than Start Date "{self.start_date}"')

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


def temporal_priority_timeline(day_of_year: int, timeline: List[str]) -> List[str]:
    """Organize the timeline according the given day of period.

    This function consists in given a determined period provided by user, the Cube builder will
    organize the timeline with the most close in time range according the reference day.

    The following example describes how it works.

    Example:
        >>> # Monthly Period - January
        >>> timeline = ['2017-01-01', '2017-01-08', '2017-01-15', '2017-01-27']
        >>> day_reference = 15
        >>> priority_timeline = temporal_priority_timeline(day_of_year=day_reference, timeline=timeline)
        >>> priority_timeline
        ... ['2017-01-15', '2017-01-08', '2017-01-27', '2017-01-01']

    Args:
        day_of_year (int) - Day refence of Period.
        timeline (List[str]) - List of timelines to be sorted.

    Returns:
        List[str] The sorted timeline values according the reference day.
    """
    if not timeline:
        return []  # TODO: Should throw exception?

    ordered_timeline = sorted(timeline)
    delta = (datetime.fromisoformat(ordered_timeline[0]) + relativedelta(days=day_of_year - 1)).date()

    def _compare_time_instant(time_instant: str):
        t = datetime.fromisoformat(time_instant).date()

        return abs(t - delta)

    output = []

    while len(ordered_timeline) > 0:
        best_time_instant = min(ordered_timeline, key=_compare_time_instant)
        ordered_timeline.pop(ordered_timeline.index(best_time_instant))
        output.append(best_time_instant)

    return output
