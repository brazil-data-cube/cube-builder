import datetime
from dateutil.relativedelta import relativedelta

from cube_builder.utils.timeline import Timeline


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

    def test_cycle_year_16days(self):
        timeline = self._build_timeline(schema='Cyclic', unit='day', step=16, cycle=dict(unit='year', step=1))

        assert len(timeline) == 23
        assert timeline[-1][-1] == datetime.date(year=2020, month=12, day=31)
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

        for time_group in timeline:
            start, end = time_group
            assert start == current_date
            current_date += relativedelta(months=3)
            assert end == current_date - relativedelta(days=1)

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

    # def test_continuous_with_interval_16d(self):
    #     timeline = self._build_timeline(
    #         schema='Continuous',
    #         unit='day',
    #         step=16,
    #         intervals=[
    #             '01-01_06-30'  # 01
    #         ],
    #         # start_date=datetime.date(year=2020, month=1, day=1),
    #         # end_date=datetime.date(year=2021, month=12, day=31),
    #     )
    #     # 2020-12-01 -> 2020-12-16
    #     # 2020-12-17 -> 2021-01-01
    #
    #     assert len(timeline) == 5
