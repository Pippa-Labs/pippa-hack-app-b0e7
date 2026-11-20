"""London Right Now: the best London venue for the time you have free.

Pippa hands this app a place, a time window and what the person is in the
mood for. It returns one venue that is open for the whole window, up to four
backups that each offer a different angle, and the facts behind each. Pippa
writes the sentence; the app only supplies facts.

How one call goes:

1. Read the inputs and settle the window (start, end, or a default length).
2. Turn the place into coordinates (postcodes.io, or Nominatim).
3. In parallel: the venues within walking reach (Google Places when its key
   is set, else OpenStreetMap) and whether rain is expected (Open-Meteo).
4. `scoring.shortlist` drops what fails a hard rule; `scoring.rank` scores the
   rest. When BestTime is set up, the top few get real busyness and are
   scored again.
5. The best is the pick; `scoring.backups` finds the alternatives.

The app keeps nothing between calls and is not told who is asking, so it has
no memory of anyone. If one source fails, the answer still comes back, with
the gap written into `assumptions`. Busyness is never guessed: with no data
it is "unknown".
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from pippa_agentapp_sdk import (
    AgentAppError,
    Call,
    Manifest,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
    load_manifest_file,
    validate_manifest,
)

import scoring
import sources
from hours import LONDON
from scoring import CATEGORIES, Candidate, Request

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")
OUTCOME = "venue.pick"

#: Nominatim refuses a request without a descriptive User-Agent.
USER_AGENT = "pippa-agent-app-london-right-now/0.1 (https://github.com/Pippa-Labs)"

#: Pippa allows 20 seconds; stop starting new requests well before that.
BUDGET_SECONDS = 16.0
#: How many of the best-scoring places get a BestTime lookup.
BUSYNESS_LOOKUPS = 6
#: A window starting within this many minutes of now asks for live busyness.
LIVE_WITHIN_MIN = 15
#: Longest window the app plans for.
MAX_WINDOW_MIN = 12 * 60

CATEGORY_WORDS = {
    "": "any", "any": "any", "anything": "any", "anywhere": "any",
    "cafe": "cafe", "café": "cafe", "cafes": "cafe", "coffee": "cafe",
    "coffee shop": "cafe", "tea": "cafe", "tea room": "cafe",
    "museum": "museum", "museums": "museum",
    "gallery": "gallery", "galleries": "gallery", "art gallery": "gallery", "art": "gallery",
    "bookshop": "bookshop", "bookstore": "bookshop", "book shop": "bookshop", "books": "bookshop",
    "restaurant": "restaurant", "restaurants": "restaurant", "food": "restaurant",
    "lunch": "restaurant", "dinner": "restaurant", "brunch": "restaurant",
}

#: Words that are not a place. The app asks where, rather than guess.
VAGUE_LOCATIONS = {"me", "here", "near me", "nearby", "around here", "my location",
                   "where i am", "london", "in london"}
#: Names Nominatim does not resolve to one point, given one.
LOCATION_ALIASES = {"central london": (51.5080, -0.1281, "Trafalgar Square")}


class PlaceNotFound(AgentAppError):
    user_message = "I could not find that place in London."
    retryable = False


class OutsideLondon(AgentAppError):
    user_message = "That place is outside London, and this app only covers London."
    retryable = False


class BadTimeWindow(AgentAppError):
    user_message = "The free time ends before it starts. Say when it starts and how long you have."
    retryable = False


class CannotReadTime(AgentAppError):
    user_message = "I could not read the start or end time. Try a time such as 2pm or 14:00."
    retryable = False


class SourceFailed(Exception):
    """A data source did not answer usefully. Internal: never shown to anyone."""

    def __init__(self, source: str, reason, retry_after: float | None = None) -> None:
        super().__init__(f"{source}: {reason}")
        self.source, self.reason, self.retry_after = source, reason, retry_after


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a key in a query string or header must reach only its provider."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class App:
    def __init__(self) -> None:
        self._manifest = validate_manifest(load_manifest_file(MANIFEST_PATH))

    @property
    def manifest(self) -> Manifest:
        return self._manifest

    # ── The one network seam ────────────────────────────────────────────────

    def _fetch(self, url: str, timeout: float = 6.0, body: bytes | None = None,
               headers: dict | None = None, secret_header: tuple[str, str] | None = None) -> bytes:
        """Every outbound request goes through here. Tests replace it."""
        request = urllib.request.Request(
            url, data=body, method="POST" if body is not None else "GET",
            headers={"User-Agent": USER_AGENT, **(headers or {})})
        if secret_header:
            # Never in headers=: those would be sent on to wherever a redirect
            # pointed. (Redirects are refused anyway.)
            request.add_unredirected_header(*secret_header)
        with _OPENER.open(request, timeout=timeout) as response:
            return response.read()

    def _get(self, source: str, url: str, timeout: float, **kwargs) -> bytes:
        """`_fetch`, with every failure turned into one internal exception.

        The provider's own error text and the URL are never kept: some
        providers repeat the key in them.
        """
        try:
            return self._fetch(url, timeout=timeout, **kwargs)
        except urllib.error.HTTPError as exc:        # first: it IS a URLError
            retry_after = None
            if exc.code == 429:
                try:
                    retry_after = float(exc.headers.get("Retry-After", ""))
                except (TypeError, ValueError, AttributeError):
                    retry_after = None
            raise SourceFailed(source, exc.code, retry_after) from None
        except TimeoutError:                         # NOT a URLError
            raise SourceFailed(source, "timeout") from None
        except (urllib.error.URLError, OSError):
            raise SourceFailed(source, "unreachable") from None

    @staticmethod
    def _timeout(deadline: float, preferred: float) -> float:
        return max(0.0, min(preferred, deadline - time.monotonic() - 0.5))

    # ── The call ────────────────────────────────────────────────────────────

    def handle(self, call: Call) -> Result:
        if call.outcome_id != OUTCOME:
            raise AgentAppError(f"this app cannot do {call.outcome_id!r}")
        deadline = time.monotonic() + BUDGET_SECONDS
        inputs = call.inputs
        now = call.context.now.astimezone(LONDON)
        assumptions: list[str] = []

        category = self._category(inputs.get("category"), assumptions)
        start, end = self._window(inputs, now, category, assumptions)

        place = str(inputs.get("location") or "").strip()
        if not place or place.lower() in VAGUE_LOCATIONS:
            return Result(
                data=_data("needs_input", assumptions, [], needs=["location"]),
                narrative="I need to know where in London you are: an area, a station or a postcode.",
            )
        origin, label, used = self._locate(place, deadline, assumptions)

        vibes = tuple(str(v).strip() for v in _as_list(inputs.get("vibe")) if str(v).strip())
        request = Request(
            category=category, origin=origin, start=start, end=end,
            cuisine=str(inputs.get("cuisine") or "").strip(),
            vibes=vibes,
            budget=self._budget(inputs.get("budget"), assumptions),
            exclude=tuple(str(x).strip() for x in _as_list(inputs.get("exclude_venues")) if str(x).strip()),
        )
        for vibe in scoring.wanted_features(vibes)[1]:
            assumptions.append(f"vibe '{vibe}': no data to match it on")
        assumptions.append(
            f"travel: walking time estimated from distance, at most {request.max_travel_min} min")

        radius = scoring.search_radius_m(request.max_travel_min)
        kinds = CATEGORIES if category == "any" else (category,)
        with ThreadPoolExecutor(max_workers=2) as pool:
            venues_job = pool.submit(self._venues, kinds, origin, radius, deadline)
            rain_job = pool.submit(self._rain, origin, start, deadline)
            venues, venue_notes, venue_source = venues_job.result()
            raining = rain_job.result()
        assumptions += venue_notes
        used.append(venue_source)
        if raining is None:
            assumptions.append("weather: unavailable")
        else:
            used.append("Open-Meteo")
            if raining:
                assumptions.append(f"weather: rain expected around {start:%H:00}")
        request = replace(request, raining=bool(raining))

        candidates = scoring.shortlist(request, venues)
        if not candidates:
            assumptions.append(
                f"checked {len(venues)} places within a {request.max_travel_min} min walk; "
                "none is open for the whole window")
            return Result(
                data=_data("no_results", assumptions, used),
                narrative=(f"Nothing within a {request.max_travel_min}-minute walk of {label} "
                           "is open for the whole of that time."),
            )

        ranked = scoring.rank(request, candidates)
        key = os.environ.get("BESTTIME_API_KEY", "")
        if key:
            ranked, note, worked = self._busyness(request, ranked, key, now, deadline)
            ranked = scoring.rank(request, ranked)
            if note:
                assumptions.append(note)
            if worked:
                used.append("BestTime")
        else:
            assumptions.append("busyness: unavailable (BestTime is not set up)")

        pick = ranked[0]
        alternatives = scoring.backups(ranked)
        return Result(
            data=_data("ok", assumptions, used, pick=pick, backups=alternatives),
            narrative=(f"{pick.venue.name} ({pick.venue.category}) is a {pick.travel_min}-minute "
                       f"walk from {label} and {scoring.open_until_text(pick.arrival, pick.closes_at)}, "
                       f"leaving {pick.time_available_min} minutes there."),
        )

    # ── Reading the inputs ──────────────────────────────────────────────────

    @staticmethod
    def _category(value, assumptions: list[str]) -> str:
        text = " ".join(str(value or "").lower().split())
        if text in CATEGORY_WORDS:
            return CATEGORY_WORDS[text]
        assumptions.append(f"category '{value}' is not covered, so every kind of place was considered")
        return "any"

    @staticmethod
    def _budget(value, assumptions: list[str]) -> int | None:
        if value is None or value == "":
            return None
        try:
            budget = int(value)
        except (TypeError, ValueError):
            budget = 0
        if 1 <= budget <= 4:
            return budget
        assumptions.append(f"budget '{value}' is not between 1 and 4, so price was ignored")
        return None

    @staticmethod
    def _window(inputs, now: datetime, category: str, assumptions: list[str]) -> tuple[datetime, datetime]:
        start = _moment(inputs.get("start"))
        if start is None:
            start = now
            assumptions.append("start: now")
        end = _moment(inputs.get("end"))
        duration = inputs.get("duration_min")
        if end is not None:
            if end <= start:
                raise BadTimeWindow(f"end {end.isoformat()} is not after start {start.isoformat()}")
        elif isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
            end = start + timedelta(minutes=int(duration))
        else:
            minutes = scoring.DEFAULT_DURATION_MIN[category]
            end = start + timedelta(minutes=minutes)
            assumptions.append(f"duration_min: {minutes} (default for {category})")
        if end - start > timedelta(minutes=MAX_WINDOW_MIN):
            end = start + timedelta(minutes=MAX_WINDOW_MIN)
            assumptions.append(f"window shortened to {MAX_WINDOW_MIN // 60} hours")
        return start, end

    # ── Sources ─────────────────────────────────────────────────────────────

    def _locate(self, place: str, deadline: float, assumptions: list[str]):
        """Coordinates for the place, a label for it, and the sources used."""
        alias = LOCATION_ALIASES.get(place.lower())
        if alias:
            assumptions.append(f"location: '{place}' taken as {alias[2]}")
            return (alias[0], alias[1]), alias[2], []

        match = sources.COORDINATES.match(place)
        if match:
            lat, lng = float(match.group(1)), float(match.group(2))
            self._check_london(place, lat, lng)
            return (lat, lng), "there", []

        compact = place.replace(" ", "")
        if sources.POSTCODE.match(place) or sources.OUTCODE.match(compact):
            try:
                found = sources.parse_postcodes(
                    self._get("postcodes.io", sources.postcodes_url(place), self._timeout(deadline, 5)))
            except SourceFailed as exc:
                if exc.reason == 404:
                    found = None
                else:
                    raise _unavailable(exc, "could not look up the postcode") from None
            except sources.BadResponse:
                raise ServiceUnavailableError("postcodes.io answered with something unreadable") from None
            if found is None:
                raise _not_found(place)
            self._check_london(place, *found)
            assumptions.append(f"location: postcode {place.upper()}")
            return found, place.upper(), ["postcodes.io"]

        try:
            found = sources.parse_nominatim(
                self._get("nominatim", sources.nominatim_url(place), self._timeout(deadline, 5)))
        except SourceFailed as exc:
            raise _unavailable(exc, "could not look up the place") from None
        except sources.BadResponse:
            raise ServiceUnavailableError("nominatim answered with something unreadable") from None
        if found is None:
            raise _not_found(place)
        lat, lng, name = found
        self._check_london(place, lat, lng)
        assumptions.append(f"location: '{place}' taken as {name[:80]}")
        return (lat, lng), place, ["OpenStreetMap Nominatim"]

    @staticmethod
    def _check_london(place: str, lat: float, lng: float) -> None:
        if not sources.in_london(lat, lng):
            error = OutsideLondon(f"{place!r} resolved to {lat:.3f},{lng:.3f}")
            error.user_message = f"{place} is outside London, and this app only covers London."
            raise error

    def _venues(self, kinds, origin, radius: int, deadline: float):
        """The venues, notes for `assumptions`, and the name of the source used."""
        notes: list[str] = []
        google_key = os.environ.get("GOOGLE_PLACES_API_KEY", "")
        if google_key:
            try:
                raw = self._get(
                    "google", sources.GOOGLE_URL, self._timeout(deadline, 6),
                    body=sources.google_body(kinds, origin, radius),
                    headers={"Content-Type": "application/json",
                             "X-Goog-FieldMask": sources.GOOGLE_FIELDS},
                    secret_header=("X-Goog-Api-Key", google_key),
                )
                venues, unreadable = sources.parse_google(raw)
                if venues:
                    if unreadable:
                        notes.append(f"venues: {unreadable} left out because Google lists no opening hours")
                    return venues, notes, "Google Places"
                notes.append("venues: Google Places found none, so OpenStreetMap was used")
            except (SourceFailed, sources.BadResponse):
                notes.append("venues: Google Places unavailable, so OpenStreetMap was used")

        failures: list[SourceFailed] = []
        for url in sources.OVERPASS_URLS:
            timeout = self._timeout(deadline, 9)
            if timeout < 2:
                break
            try:
                raw = self._get(
                    "overpass", url, timeout,
                    body=sources.overpass_body(kinds, origin, radius, max(2, int(timeout) - 1)),
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                venues, unreadable = sources.parse_overpass(raw)
            except SourceFailed as exc:
                failures.append(exc)
                continue
            except sources.BadResponse:
                failures.append(SourceFailed("overpass", "unreadable"))
                continue
            notes.append("venues: only places with published opening hours are considered")
            if unreadable:
                notes.append(f"venues: {unreadable} left out because their opening hours could not be read")
            return venues, notes, "OpenStreetMap"

        limited = next((f for f in failures if f.reason == 429), None)
        if limited:
            raise RateLimitedError("overpass rate limited", retry_after_seconds=limited.retry_after)
        raise ServiceUnavailableError(
            "no venue source answered: " + ", ".join(str(f.reason) for f in failures))

    def _rain(self, origin, when: datetime, deadline: float) -> bool | None:
        timeout = self._timeout(deadline, 4)
        if timeout < 1:
            return None
        try:
            return sources.parse_rain(
                self._get("open-meteo", sources.open_meteo_url(origin, when), timeout), when)
        except (SourceFailed, sources.BadResponse, LookupError, TypeError, ValueError, AttributeError):
            # Weather only nudges the score, so an odd reply is reported, not fatal.
            return None

    def _busyness(self, request: Request, ranked: list[Candidate], key: str,
                  now: datetime, deadline: float):
        """Busyness for the best few. Returns the candidates, a note, and whether any worked."""
        live = abs((request.start - now).total_seconds()) <= LIVE_WITHIN_MIN * 60
        top = ranked[:BUSYNESS_LOOKUPS]

        def lookup(candidate: Candidate):
            venue = candidate.venue
            if not venue.address:
                return None
            address = venue.address if "london" in venue.address.lower() else f"{venue.address}, London"
            timeout = self._timeout(deadline, 6)
            if timeout < 1.5:
                return None
            try:
                raw = self._get("besttime", sources.besttime_url(live, key, venue.name, address),
                                timeout, body=b"")
                if live:
                    return sources.parse_besttime_live(raw)
                return sources.parse_besttime_forecast(raw, candidate.arrival, request.end)
            except (SourceFailed, sources.BadResponse, LookupError, TypeError, ValueError, AttributeError):
                return None     # busyness stays "unknown", never guessed

        with ThreadPoolExecutor(max_workers=len(top) or 1) as pool:
            found = list(pool.map(lookup, top))

        updated = {}
        for candidate, busyness in zip(top, found):
            if busyness is not None and busyness.level != "unknown":
                updated[candidate.venue.venue_id] = busyness
        missing = len(top) - len(updated)
        note = ""
        if missing:
            note = f"busyness: unavailable for {missing} of the {len(top)} best places"
        if len(ranked) > len(top):
            note = (note + "; " if note else "busyness: ") + f"only the {len(top)} best places were checked"
        result = [replace(c, busyness=updated[c.venue.venue_id]) if c.venue.venue_id in updated else c
                  for c in ranked]
        return result, note, bool(updated)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [part for part in str(value).split(",")]


def _moment(value) -> datetime | None:
    """A datetime input, which may arrive parsed or as ISO text. Naive means London."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            raise CannotReadTime(f"unreadable time {value!r}") from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=LONDON)
    return moment.astimezone(LONDON)


def _not_found(place: str) -> PlaceNotFound:
    error = PlaceNotFound(f"no match for {place!r}")
    error.user_message = f"I could not find {place} in London. Try an area, a station or a postcode."
    return error


def _unavailable(exc: SourceFailed, message: str) -> AgentAppError:
    if exc.reason == 429:
        return RateLimitedError(f"{exc.source} rate limited", retry_after_seconds=exc.retry_after)
    return ServiceUnavailableError(f"{message}: {exc.source} {exc.reason}")


def _backup_line(angle: str, c: Candidate) -> str:
    line = (f"{angle}: {c.venue.name} ({c.venue.category}), {c.travel_min} min walk, "
            f"{scoring.open_until_text(c.arrival, c.closes_at)}, busyness {c.busyness.level}")
    if c.venue.rating is not None:
        line += f", rating {c.venue.rating:g}"
    return line + f", id {c.venue.venue_id}"


def _data(status: str, assumptions: list[str], used: list[str], *, needs=(),
          pick: Candidate | None = None, backups=()) -> dict:
    data = {
        "status": status,
        "needs": list(needs),
        "assumptions": list(assumptions),
        "sources": list(dict.fromkeys(used)),
        "backups": [_backup_line(angle, c) for angle, c in backups],
    }
    if pick is None:
        return data
    venue = pick.venue
    data.update({
        "pick_venue_id": venue.venue_id,
        "pick_name": venue.name,
        "pick_category": venue.category,
        "pick_lat": round(venue.lat, 6),
        "pick_lng": round(venue.lng, 6),
        "pick_maps_url": venue.maps_url,
        "pick_travel_mode": "walk",
        "pick_travel_min": pick.travel_min,
        "pick_open_until": scoring.open_until_text(pick.arrival, pick.closes_at).removeprefix("open until "),
        "pick_time_available_min": pick.time_available_min,
        "pick_busyness": pick.busyness.level,
        "pick_busyness_source": pick.busyness.source,
        "pick_reasons": list(pick.reasons),
        "pick_score": round(pick.score, 2),
    })
    optional = {
        "pick_address": venue.address,
        "pick_busier_after": pick.busyness.busier_after,
        "pick_rating": round(venue.rating, 1) if venue.rating is not None else None,
        "pick_review_count": venue.review_count,
        "pick_price_level": venue.price_level,
        "pick_tip": venue.tip,
    }
    data.update({name: value for name, value in optional.items() if value is not None})
    return data
