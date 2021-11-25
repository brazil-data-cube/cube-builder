#
# This file is part of Python Module for Cube Builder.
# Copyright (C) 2019-2021 INPE.
#
# Cube Builder is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#

"""Define the unittests for data cube timeline."""

import datetime

import pytest
from dateutil.relativedelta import relativedelta

from cube_builder.utils.timeline import Timeline, temporal_priority_timeline


class TestTimeline:
    start_date = datetime.date(year=2020, month=1, day=1)
    end_date = datetime.date(year=2020, month=12, day=31)

    def _build_timeline(self, schema, unit, step, cycle=None, intervals=None, start_date=None, end_date=None):
        return Timeline(
            schema=schema,
            start_date=start_date or self.start_date,
            end_date=end_date or self.end_date,
            unit=unit,
            step=step,
            cycle=cycle,
            intervals=intervals
        ).mount()

    @staticmethod
    def _assert_interval(ref, delta, timeline):
        for begin, end in timeline:
            assert ref == begin
            ref += delta - relativedelta(days=1)
            assert end == ref
            ref += relativedelta(days=1)

    def test_continuous_step_month(self):
        timeline = self._build_timeline(schema='Continuous', unit='month', step=1)

        assert len(timeline) == 12

        for i, (time_inst_start, time_inst_end) in enumerate(timeline):
            expected_begin = (self.start_date + relativedelta(months=i))
            expected_end = (self.start_date + relativedelta(months=i+1) - relativedelta(days=1))
            assert time_inst_start == expected_begin
            assert time_inst_end == expected_end

    def test_continuous_step_day(self):
        timeline = self._build_timeline(schema='Continuous', unit='day', step=16)

        assert len(timeline) == 23

        delta = relativedelta(days=16)

        ref = self.start_date

        for i, (time_inst_start, time_inst_end) in enumerate(timeline):
            assert time_inst_start == ref

            ref += delta

            assert time_inst_end == ref - relativedelta(days=1)

        assert timeline[-1][-1].year == 2021

    def test_continuous_step_day_start06(self):
        expected = datetime.date(year=2020, month=6, day=12)
        timeline = self._build_timeline(
            schema='Continuous',
            unit='day',
            step=16,
            start_date=expected,
            end_date=self.end_date
        )

        assert len(timeline) == 13
        assert timeline[0][0] == expected  # Continuous should start same day of start_date
        assert timeline[-1][-1].year == 2021

        delta = relativedelta(days=16)

        self._assert_interval(expected, delta, timeline)

    def test_cycle_year_16days(self):
        timeline = self._build_timeline(schema='Cyclic', unit='day', step=16, cycle=dict(unit='year', step=1))

        assert len(timeline) == 23
        assert timeline[-1][-1] == datetime.date(year=2020, month=12, day=31)

        delta = relativedelta(days=16)
        expected = datetime.date(year=self.start_date.year, month=self.start_date.month, day=self.start_date.day)

        self._assert_interval(expected, delta, timeline[:-1])

        assert (timeline[-1][-1] - timeline[-1][0]).days < 16  # Last period should be less than 16 days

    def test_cycle_year_16days_starting_half(self):
        timeline = self._build_timeline(
            schema='Cyclic',
            unit='day',
            step=16,
            cycle=dict(unit='year', step=1),
            start_date=datetime.date(year=2020, month=6, day=15),
            end_date=self.end_date
        )

        assert len(timeline) == 12

        delta = relativedelta(days=16)
        expected = datetime.date(year=2020, month=6, day=25)
        for start, end in timeline[:-1]:
            assert start == expected
            expected += delta
            assert end == (expected - relativedelta(days=1))

        assert timeline[-1][-1] == datetime.date(year=2020, month=12, day=31)
        assert (timeline[-1][-1] - timeline[-1][0]).days < 16  # Last period should be less than 16 days

    def test_cycle_3month(self):
        timeline = self._build_timeline(
            schema='Cyclic',
            unit='month',
            cycle=dict(
                unit='year',
                step=1
            ),
            step=3,
        )

        assert len(timeline) == 4

        current_date = self.start_date

        self._assert_interval(current_date, relativedelta(months=3), timeline)

    def test_cycle_with_interval(self):
        timeline = self._build_timeline(
            schema='Cyclic',
            unit='month',
            step=3,
            cycle=dict(
                unit='year',
                step=1,
                intervals=[
                    '08-01_10-31',
                ]
            ),
            start_date=datetime.date(year=2000, month=1, day=1),
            end_date=datetime.date(year=2002, month=12, day=31),
        )
        assert len(timeline) == 3
        expected_begin = datetime.date(year=2000, month=8, day=1)
        for time_group in timeline:
            start, end = time_group

            expected_begin = expected_begin.replace(year=start.year)

            assert start == expected_begin
            expected_end = expected_begin + (relativedelta(months=3) - relativedelta(days=1))
            assert end == expected_end

    def test_continuous_with_interval_season(self):
        timeline = self._build_timeline(
            schema='Continuous',
            unit='month',
            step=3,
            intervals=[
                '12-21_03-20',
                '03-21_06-20',
                '06-21_09-21',
                '09-22_12-20'
            ]
        )

        assert len(timeline) == 5
        # Should match first time instant with previous year
        assert timeline[0][0] == datetime.date(year=2019, month=12, day=21)
        # Should match last time instant with next year
        assert timeline[-1][-1] == datetime.date(year=2021, month=3, day=20)

    def test_invalid_date_limit(self):
        start = datetime.datetime(year=2021, month=9, day=30)
        end = datetime.datetime(year=2020, month=10, day=15)
        with pytest.raises(ValueError) as e:
            self._build_timeline(schema='Continuous', unit='month', step=1, start_date=start, end_date=end)
        assert ' must not be lower than Start Date ' in str(e.value)


class TestTemporalPriorityTimeline:
    """Test the generation of timeline using Temporal Priority Timeline algorithm."""

    @staticmethod
    def _assert_timeline(given: list, expected_list: list):
        assert len(given) == len(expected_list)

        for idx, time_instant in enumerate(given):
            expected_time_instant = expected_list[idx]

            assert time_instant == expected_time_instant

    def test_day15_monthly(self):
        timeline = [
            '2020-01-01',
            '2020-01-07',
            '2020-01-12',
            '2020-01-15',
            '2020-01-18',
            '2020-01-23',
            '2020-01-28',
            '2020-02-01',
        ]
        # We refers the day 31 to be the last day of month period
        reference_day = 15
        ts = temporal_priority_timeline(reference_day, timeline)

        expected = [
            '2020-01-15',
            '2020-01-12',
            '2020-01-18',
            '2020-01-07',
            '2020-01-23',
            '2020-01-28',
            '2020-01-01',
            '2020-02-01',
        ]

        self._assert_timeline(ts, expected)

    def test_last_day_of_period(self):
        timeline = [
            '2020-12-12',
            '2020-12-15',
            '2020-12-01',
            '2020-12-07',
            '2020-12-18',
            '2020-12-28',
            '2020-12-23',
            '2021-01-01',
        ]
        # We refers the day 31 to be the last day of month period
        reference_day = 31
        ts = temporal_priority_timeline(reference_day, timeline)
        ordered_desc = sorted([datetime.datetime.fromisoformat(t) for t in timeline], reverse=True)
        expected = [elm.strftime('%Y-%m-%d') for elm in ordered_desc]

        self._assert_timeline(ts, expected)

    def test_day42_a_quarter(self):
        timeline = [
            '2019-04-01',
            '2019-04-18',
            '2019-04-25',
            '2019-05-03',
            '2019-05-31',
            '2019-06-13',
        ]
        reference = 42
        ts = temporal_priority_timeline(reference, timeline)
        # TODO: 05-03   04-25    05-31  04-18   06-13   04-01
        expected = ['2019-05-03', '2019-04-25', '2019-05-31', '2019-04-18', '2019-06-13', '2019-04-01']
        self._assert_timeline(ts, expected)

    def test_timeline_16days_year_cycle(self):
        timeline = [
            '2019-12-19',
            '2019-12-20',
            '2019-12-25',
            '2019-12-26',
        ]
        reference = 2
        ts = temporal_priority_timeline(reference, timeline)

        expected = ['2019-12-20', '2019-12-19', '2019-12-25', '2019-12-26']

        self._assert_timeline(ts, expected)
