"""Opening hours: one weekly shape for every data source, and one question.

OpenStreetMap writes hours as text ("Mo-Fr 08:00-18:00; Sa 09:00-17:00") and
Google as a list of open/close points. Both are turned into `WeeklyHours`, and
the app only ever asks it one thing: "if somebody arrives at A and stays until
B, is the place open the whole time, and when does it close?"

Anything the parser does not fully understand (month ranges, public-holiday
rules that are the only rule, "sunrise", comments) gives `None`, meaning
"hours unknown", and the app leaves that place out. Guessing that a place is
open is the one mistake this app must not make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
DAY_MINUTES = 24 * 60
DAY_CODES = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")

Span = tuple[int, int]


@dataclass(frozen=True)
class WeeklyHours:
    """Opening spans for each weekday, Monday first.

    A span is (opens, closes) in minutes after that day's midnight. `closes`
    can pass 1440 when a place shuts after midnight: (1080, 1560) is
    18:00 to 02:00 the next morning.
    """

    days: tuple[tuple[Span, ...], ...]

    @classmethod
    def always(cls) -> "WeeklyHours":
        return cls(tuple(((0, DAY_MINUTES),) for _ in range(7)))

    def spans_between(self, first: date, last: date) -> list[tuple[datetime, datetime]]:
        """Every opening span that starts on a day from `first` to `last`, merged."""
        spans = []
        day = first
        while day <= last:
            midnight = datetime(day.year, day.month, day.day, tzinfo=LONDON)
            for opens, closes in self.days[day.weekday()]:
                spans.append((midnight + timedelta(minutes=opens),
                              midnight + timedelta(minutes=closes)))
            day += timedelta(days=1)
        spans.sort()
        merged: list[tuple[datetime, datetime]] = []
        for opens, closes in spans:
            if merged and opens <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], closes))
            else:
                merged.append((opens, closes))
        return merged

    def open_until(self, arrival: datetime, end: datetime) -> datetime | None:
        """When the place closes, if it is open from `arrival` right through `end`.

        None when it is closed at any point of that window.
        """
        arrival = arrival.astimezone(LONDON)
        end = end.astimezone(LONDON)
        # Two days back, for a span that started yesterday (or the day before)
        # and runs past midnight; one day on, so a close tomorrow is seen.
        for opens, closes in self.spans_between(arrival.date() - timedelta(days=2),
                                                end.date() + timedelta(days=1)):
            if opens <= arrival < closes:
                return closes if closes >= end else None
        return None


# ── OpenStreetMap `opening_hours` ───────────────────────────────────────────

_SPAN = re.compile(r"^(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})$")
_DAY = re.compile(r"^(Mo|Tu|We|Th|Fr|Sa|Su)(?:-(Mo|Tu|We|Th|Fr|Sa|Su))?$")
_SELECTOR = re.compile(r"^[A-Za-z][A-Za-z,\-]*$")
# "Mo-Fr 09:00-17:00, Sa 10:00-14:00": a comma followed by a day starts an
# additional rule, not another time span.
_ADDITIONAL_RULE = re.compile(r",\s*(?=(?:Mo|Tu|We|Th|Fr|Sa|Su|PH|SH)\b)")


def parse_osm(text: str | None) -> WeeklyHours | None:
    """The common subset of OpenStreetMap's opening_hours syntax, or None."""
    text = (text or "").strip()
    if not text:
        return None
    if text == "24/7":
        return WeeklyHours.always()

    week: list[list[Span]] = [[] for _ in range(7)]
    for group in text.split(";"):
        group = group.strip()
        if not group:
            continue
        for index, rule in enumerate(_ADDITIONAL_RULE.split(group)):
            parsed = _parse_rule(rule.strip())
            if parsed is None:
                return None
            days, spans = parsed
            for day in days:
                if index == 0:
                    week[day] = list(spans)   # a new ";" rule replaces those days
                else:
                    week[day].extend(spans)   # an additional "," rule adds to them
    return WeeklyHours(tuple(tuple(sorted(day)) for day in week))


def _parse_rule(rule: str) -> tuple[list[int], list[Span]] | None:
    if rule == "24/7":
        return list(range(7)), [(0, DAY_MINUTES)]

    selector, _, rest = rule.partition(" ")
    if _SELECTOR.match(selector) and selector.lower() not in ("off", "closed"):
        days = _parse_days(selector)
        if days is None:
            return None
        times = rest.strip()
    else:
        days, times = list(range(7)), rule

    spans = _parse_times(times)
    if spans is None:
        return None
    return days, spans


def _parse_days(selector: str) -> list[int] | None:
    days: list[int] = []
    for part in selector.split(","):
        if part in ("PH", "SH"):
            # Public and school holidays: no calendar of them here, so the
            # rule is applied to ordinary days only. A rule that is ONLY about
            # holidays ends up applying to no day at all.
            continue
        match = _DAY.match(part)
        if not match:
            return None
        first = DAY_CODES.index(match.group(1))
        last = DAY_CODES.index(match.group(2) or match.group(1))
        day = first
        while True:
            if day not in days:
                days.append(day)
            if day == last:
                break
            day = (day + 1) % 7      # "Sa-Mo" wraps round the week
    return days


def _parse_times(times: str) -> list[Span] | None:
    if times.lower() in ("off", "closed"):
        return []
    if not times:
        return None
    spans: list[Span] = []
    for part in times.split(","):
        match = _SPAN.match(part.strip())
        if not match:
            return None
        h1, m1, h2, m2 = (int(group) for group in match.groups())
        if h1 > 24 or h2 > 48 or m1 > 59 or m2 > 59:
            return None
        opens, closes = h1 * 60 + m1, h2 * 60 + m2
        if closes <= opens:
            closes += DAY_MINUTES      # "18:00-02:00" shuts the next morning
        spans.append((opens, closes))
    return spans


# ── Google Places `regularOpeningHours.periods` ─────────────────────────────


def from_google_periods(periods) -> WeeklyHours | None:
    """Google's open/close points, where day 0 is SUNDAY, as WeeklyHours."""
    if not isinstance(periods, list):
        return None
    week: list[list[Span]] = [[] for _ in range(7)]
    for period in periods:
        if not isinstance(period, dict) or not isinstance(period.get("open"), dict):
            continue
        point = period["open"]
        close = period.get("close")
        if close is None:
            # Google's way of saying "always open": one open point, no close.
            return WeeklyHours.always()
        day = (int(point.get("day", 0)) - 1) % 7          # Sunday 0 -> Monday 0
        opens = int(point.get("hour", 0)) * 60 + int(point.get("minute", 0))
        days_on = (int(close.get("day", 0)) - int(point.get("day", 0))) % 7
        closes = days_on * DAY_MINUTES + int(close.get("hour", 0)) * 60 + int(close.get("minute", 0))
        if closes <= opens:
            closes += DAY_MINUTES
        week[day].append((opens, closes))
    return WeeklyHours(tuple(tuple(sorted(day)) for day in week))
