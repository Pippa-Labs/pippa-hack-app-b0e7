"""Opening hours: the one part where a mistake sends somebody to a locked door."""

from __future__ import annotations

from datetime import datetime

import pytest

from hours import LONDON, WeeklyHours, from_google_periods, parse_osm


def at(day: int, hour: int, minute: int = 0) -> datetime:
    """A moment in November 2026, London time. 16 November is a Monday."""
    return datetime(2026, 11, day, hour, minute, tzinfo=LONDON)


# ── OpenStreetMap text ──────────────────────────────────────────────────────


def test_weekdays_and_saturday():
    hours = parse_osm("Mo-Fr 08:00-18:00; Sa 09:00-17:00")
    assert hours.days[0] == ((480, 1080),)
    assert hours.days[4] == ((480, 1080),)
    assert hours.days[5] == ((540, 1020),)
    assert hours.days[6] == ()


def test_always_open():
    assert parse_osm("24/7") == WeeklyHours.always()


def test_a_later_rule_replaces_earlier_days():
    hours = parse_osm("Mo-Su 10:00-18:00; We 10:00-21:00")
    assert hours.days[1] == ((600, 1080),)
    assert hours.days[2] == ((600, 1260),)


def test_a_comma_and_a_day_adds_a_rule():
    hours = parse_osm("Mo-Fr 09:00-17:00, Sa 10:00-14:00")
    assert hours.days[0] == ((540, 1020),)
    assert hours.days[5] == ((600, 840),)


def test_two_spans_in_one_day():
    hours = parse_osm("Mo-Su 11:30-15:00,17:00-23:00")
    assert hours.days[3] == ((690, 900), (1020, 1380))


def test_a_day_range_wraps_round_the_week():
    hours = parse_osm("Su-Tu 10:00-12:00")
    assert hours.days[6] and hours.days[0] and hours.days[1]
    assert hours.days[2] == ()


def test_off_closes_a_day():
    hours = parse_osm("Mo-Sa 09:00-17:00; Mo off")
    assert hours.days[0] == ()
    assert hours.days[1] == ((540, 1020),)


def test_a_holiday_rule_leaves_ordinary_days_alone():
    hours = parse_osm("Mo-Fr 09:00-17:00; PH off")
    assert hours.days[0] == ((540, 1020),)


@pytest.mark.parametrize("text", [
    "", None, "sunrise-sunset", "Jan-Mar Mo-Fr 09:00-17:00", "Mo-Fr 9am-5pm",
    "by appointment",
])
def test_anything_not_understood_is_unknown(text):
    """None means 'hours unknown', and the app leaves the place out."""
    assert parse_osm(text) is None


# ── Is it open for the whole visit? ─────────────────────────────────────────


def test_open_through_the_window_gives_the_closing_time():
    hours = parse_osm("Mo-Su 09:00-17:00")
    assert hours.open_until(at(18, 16, 0), at(18, 16, 55)) == at(18, 17, 0)


def test_closing_before_the_end_is_not_open():
    hours = parse_osm("Mo-Su 09:00-17:00")
    assert hours.open_until(at(18, 16, 30), at(18, 17, 30)) is None


def test_arriving_before_it_opens_is_not_open():
    hours = parse_osm("Mo-Su 09:00-17:00")
    assert hours.open_until(at(18, 8, 30), at(18, 10, 0)) is None


def test_a_lunchtime_gap_counts_as_closed():
    hours = parse_osm("Mo-Su 11:30-15:00,17:00-23:00")
    assert hours.open_until(at(18, 14, 30), at(18, 15, 30)) is None
    assert hours.open_until(at(18, 17, 30), at(18, 19, 0)) == at(18, 23, 0)


def test_past_midnight_belongs_to_the_day_before():
    hours = parse_osm("Mo-Su 18:00-02:00")
    # 01:00 on Tuesday is still Monday night's opening.
    assert hours.open_until(at(17, 1, 0), at(17, 1, 30)) == at(17, 2, 0)


def test_closed_day_is_closed():
    hours = parse_osm("Tu-Su 10:00-18:00")
    assert hours.open_until(at(16, 12, 0), at(16, 13, 0)) is None     # a Monday


# ── Google's open/close points ──────────────────────────────────────────────


def test_google_days_start_on_sunday():
    hours = from_google_periods([
        {"open": {"day": 1, "hour": 9, "minute": 0}, "close": {"day": 1, "hour": 17, "minute": 0}},
        {"open": {"day": 0, "hour": 11, "minute": 0}, "close": {"day": 0, "hour": 16, "minute": 0}},
    ])
    assert hours.days[0] == ((540, 1020),)      # Monday
    assert hours.days[6] == ((660, 960),)       # Sunday


def test_google_overnight_period():
    hours = from_google_periods([
        {"open": {"day": 5, "hour": 18, "minute": 0}, "close": {"day": 6, "hour": 2, "minute": 0}},
        {"open": {"day": 6, "hour": 20, "minute": 0}, "close": {"day": 0, "hour": 1, "minute": 0}},
    ])
    assert hours.days[4] == ((1080, 1560),)     # Friday 18:00 to 02:00
    assert hours.days[5] == ((1200, 1500),)     # Saturday 20:00 to 01:00


def test_google_always_open():
    assert from_google_periods([{"open": {"day": 0, "hour": 0, "minute": 0}}]) == WeeklyHours.always()


def test_google_no_hours_is_unknown():
    assert from_google_periods(None) is None
