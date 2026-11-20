"""Choosing the venue: the data model, hard filters, the score and the backups.

Pure functions over plain data. Nothing here touches the network or reads the
clock, so the same request and the same venues always give the same answer,
and every rule can be tested on its own.

The order of work:

1. `shortlist`  drops every venue that fails a hard rule: wrong kind of place,
                turned down, a stated need it is known not to meet, too far,
                or not open for the whole of the time the person would be
                there (from arrival, which is start plus travel, to the end).
2. `rank`       scores the rest from 0 to 1 and sorts them, best first.
3. `backups`    picks up to four alternatives, each with its own angle.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from hours import LONDON, WeeklyHours

# ── Configuration ───────────────────────────────────────────────────────────

#: How much each part of the score counts. The brief's fifth weight, novelty
#: (0.15), needs a memory of where this person has been, and a Pippa app has
#: no memory and is never told who is asking. So it is left out, and the four
#: below are divided by their total, which keeps the score between 0 and 1.
WEIGHTS = {"preference": 0.35, "timing": 0.20, "proximity": 0.20, "quality": 0.10}

#: When it is raining around the time of arrival, a short walk counts for more.
RAIN_PROXIMITY_BONUS = 0.10

#: Travel may take at most this share of the window, and never more than the cap.
TRAVEL_SHARE = 0.4
TRAVEL_CAP_MIN = 30

#: Walking: 80 m a minute (4.8 km/h), and streets are longer than a straight line.
WALK_METRES_PER_MIN = 80
DETOUR_FACTOR = 1.3

#: How a busyness level scores, and how quiet counts when someone asked for quiet.
BUSYNESS_SCORE = {"quiet": 1.0, "moderate": 0.7, "busy": 0.2, "unknown": 0.5}
QUIET_FIT = {"quiet": 1.0, "moderate": 0.4, "busy": 0.0, "unknown": 0.3}

#: A place closing this long after the person leaves gets the full closing score.
CLOSING_BUFFER_FULL_MIN = 60

#: Ratings are pulled towards this average until a place has enough reviews,
#: so a 5.0 from three reviews does not beat a 4.6 from three thousand.
RATING_PRIOR = 4.2
RATING_PRIOR_WEIGHT = 50
RATING_FLOOR, RATING_CEILING = 3.5, 5.0

#: Minutes, when neither an end nor a duration was given.
DEFAULT_DURATION_MIN = {"cafe": 45, "museum": 120, "gallery": 120,
                        "restaurant": 75, "bookshop": 45, "any": 60}

CATEGORIES = ("cafe", "museum", "gallery", "bookshop", "restaurant")

MAX_BACKUPS = 4
ANGLES = ("closer", "quieter", "cheaper", "higher_rated", "open_later", "next_best")
#: Two places this close together, or with the same name, are near-copies.
NEAR_COPY_METRES = 40
OPEN_LATER_MIN = 60

#: Words in a vibe, and the venue feature they ask for. "quiet" is answered
#: from busyness rather than from a tag.
VIBE_FEATURES = {
    "wifi": "wifi", "wi-fi": "wifi", "laptop": "wifi", "work": "wifi",
    "outdoor": "outdoor_seating", "outside": "outdoor_seating",
    "terrace": "outdoor_seating", "garden": "outdoor_seating",
    "wheelchair": "wheelchair", "accessible": "wheelchair", "step-free": "wheelchair",
    "vegan": "vegan", "vegetarian": "vegetarian", "veggie": "vegetarian",
    "gluten": "gluten_free",
    "dog": "dogs",
    "kid": "children", "child": "children", "family": "children",
    "quiet": "quiet", "calm": "quiet", "peaceful": "quiet",
}
FEATURE_WORDS = {
    "wifi": "wifi", "outdoor_seating": "outdoor seating", "wheelchair": "wheelchair access",
    "vegan": "vegan options", "vegetarian": "vegetarian options",
    "gluten_free": "gluten-free options", "dogs": "dogs allowed",
    "children": "good for children", "quiet": "quiet",
}
#: A stated need: a place known NOT to meet it is dropped, not just scored down.
NEEDS = frozenset({"wheelchair", "vegan", "vegetarian", "gluten_free"})

#: What a dish or drink means in the cuisine tags the sources use.
CUISINE_TERMS = {
    "pasta": ("italian", "pasta"), "pizza": ("pizza", "italian"),
    "sushi": ("sushi", "japanese"), "ramen": ("ramen", "japanese", "noodle"),
    "noodles": ("noodle", "ramen", "chinese", "japanese", "thai", "vietnamese"),
    "curry": ("indian", "curry", "thai"), "dim sum": ("chinese", "dim_sum", "cantonese"),
    "tacos": ("mexican", "tacos"), "burger": ("burger", "american"),
    "pho": ("vietnamese",), "dumplings": ("chinese", "dumpling"),
    "coffee": ("coffee_shop", "coffee"), "flat white": ("coffee_shop", "coffee"),
    "espresso": ("coffee_shop", "coffee"), "tea": ("tea",), "cake": ("cake",),
    "brunch": ("brunch", "breakfast"), "breakfast": ("breakfast", "brunch"),
}

# ── Data model ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Venue:
    """One place, as any data source describes it."""

    venue_id: str
    name: str
    category: str
    lat: float
    lng: float
    hours: WeeklyHours
    maps_url: str
    address: str | None = None
    rating: float | None = None
    review_count: int | None = None
    price_level: int | None = None
    cuisines: tuple[str, ...] = ()
    has: frozenset[str] = frozenset()      # features the source says it has
    lacks: frozenset[str] = frozenset()    # features the source says it lacks
    tip: str | None = None


@dataclass(frozen=True)
class Busyness:
    level: str = "unknown"        # quiet | moderate | busy | unknown
    source: str = "none"          # live | forecast | none
    busier_after: str | None = None


UNKNOWN = Busyness()


@dataclass(frozen=True)
class Request:
    """What was asked, after the app has resolved the place and the window."""

    category: str
    origin: tuple[float, float]
    start: datetime
    end: datetime
    cuisine: str = ""
    vibes: tuple[str, ...] = ()
    budget: int | None = None
    exclude: tuple[str, ...] = ()
    raining: bool = False

    @property
    def duration_min(self) -> int:
        return int((self.end - self.start).total_seconds() // 60)

    @property
    def max_travel_min(self) -> int:
        return max(1, min(TRAVEL_CAP_MIN, math.floor(self.duration_min * TRAVEL_SHARE)))


@dataclass(frozen=True)
class Candidate:
    venue: Venue
    travel_min: int
    arrival: datetime
    closes_at: datetime
    time_available_min: int
    busyness: Busyness = UNKNOWN
    score: float = 0.0
    reasons: tuple[str, ...] = ()


# ── Distance ────────────────────────────────────────────────────────────────


def distance_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lng1, lat2, lng2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lng2 - lng1) / 2) ** 2)
    return 2 * 6_371_000 * math.asin(math.sqrt(h))


def walk_minutes(origin: tuple[float, float], lat: float, lng: float) -> int:
    metres = distance_m(origin, (lat, lng)) * DETOUR_FACTOR
    return max(1, math.ceil(metres / WALK_METRES_PER_MIN))


def search_radius_m(max_travel_min: int) -> int:
    """The straight-line radius that a walk of `max_travel_min` can cover."""
    return int(max_travel_min * WALK_METRES_PER_MIN / DETOUR_FACTOR)


# ── Reading the request ─────────────────────────────────────────────────────


def wanted_features(vibes) -> tuple[list[str], list[str]]:
    """The features the vibes ask for, and the vibes with no data to match."""
    wanted: list[str] = []
    unmatched: list[str] = []
    for vibe in vibes:
        text = vibe.lower()
        found = [VIBE_FEATURES[word] for word in sorted(VIBE_FEATURES) if word in text]
        if not found:
            unmatched.append(vibe)
        for feature in found:
            if feature not in wanted:
                wanted.append(feature)
    return wanted, unmatched


def cuisine_terms(cuisine: str) -> tuple[str, ...]:
    key = cuisine.strip().lower()
    return tuple(dict.fromkeys((key.replace(" ", "_"), key, *CUISINE_TERMS.get(key, ()))))


def same_name(a: str, b: str) -> bool:
    return _normalise(a) == _normalise(b)


def _normalise(name: str) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", name.lower().replace("&", "and"))
    return re.sub(r"^the ", "", " ".join(name.split()))


# ── 1. Hard filters ─────────────────────────────────────────────────────────


def shortlist(request: Request, venues) -> list[Candidate]:
    """Every venue that passes the hard rules, with its travel and closing time."""
    wanted, _ = wanted_features(request.vibes)
    needs = NEEDS.intersection(wanted)
    seen: set[str] = set()
    out: list[Candidate] = []
    for venue in venues:
        if venue.venue_id in seen:
            continue
        seen.add(venue.venue_id)
        if request.category != "any" and venue.category != request.category:
            continue
        if any(venue.venue_id == x or same_name(venue.name, x) for x in request.exclude):
            continue
        if needs & venue.lacks:
            continue
        travel = walk_minutes(request.origin, venue.lat, venue.lng)
        if travel > request.max_travel_min:
            continue
        arrival = request.start + timedelta(minutes=travel)
        if arrival >= request.end:
            continue
        closes_at = venue.hours.open_until(arrival, request.end)
        if closes_at is None:
            continue
        out.append(Candidate(
            venue=venue, travel_min=travel, arrival=arrival, closes_at=closes_at,
            time_available_min=int((request.end - arrival).total_seconds() // 60),
        ))
    return out


# ── 2. The score ────────────────────────────────────────────────────────────


def rank(request: Request, candidates) -> list[Candidate]:
    """Score every candidate and sort them, best first. Ties break on travel, then id."""
    wanted, _ = wanted_features(request.vibes)
    scored = [_score(request, candidate, wanted) for candidate in candidates]
    return sorted(scored, key=lambda c: (-c.score, c.travel_min, c.venue.venue_id))


def _score(request: Request, c: Candidate, wanted: list[str]) -> Candidate:
    venue, level = c.venue, c.busyness.level
    reasons: list[str] = []

    # Preference: the average of whichever of these the person asked about.
    parts: list[float] = []
    matched: list[str] = []
    if request.cuisine:
        if any(term in venue.cuisines for term in cuisine_terms(request.cuisine)):
            parts.append(1.0)
            matched.append(request.cuisine.lower())
        else:
            parts.append(0.0 if venue.cuisines else 0.3)
    tagged = [f for f in wanted if f != "quiet"]
    if tagged:
        hits = [f for f in tagged if f in venue.has]
        parts.append(len(hits) / len(tagged))
        matched += [FEATURE_WORDS[f] for f in hits]
    if "quiet" in wanted:
        parts.append(QUIET_FIT[level])
        if level == "quiet":
            matched.append("quiet")
    if request.budget is not None:
        if venue.price_level is None:
            parts.append(0.5)
        elif venue.price_level <= request.budget:
            parts.append(1.0)
            reasons.append("within budget")
        else:
            parts.append(max(0.0, 1 - 0.5 * (venue.price_level - request.budget)))
            reasons.append("above budget")
    preference = sum(parts) / len(parts) if parts else 0.5
    if matched:
        reasons.insert(0, "matches: " + ", ".join(matched))

    # Timing: how busy it is, and how long after the person leaves it closes.
    buffer_min = (c.closes_at - request.end).total_seconds() / 60
    timing = 0.6 * BUSYNESS_SCORE[level] + 0.4 * min(1.0, buffer_min / CLOSING_BUFFER_FULL_MIN)
    if level != "unknown":
        when = "now" if c.busyness.source == "live" else f"around {c.arrival:%H:%M} (forecast)"
        phrase = f"{level} {when}"
        if c.busyness.busier_after:
            phrase += f", busier after {c.busyness.busier_after}"
        reasons.append(phrase)
    reasons.append(open_until_text(c.arrival, c.closes_at))

    # Proximity.
    proximity = max(0.0, 1 - c.travel_min / max(1, request.max_travel_min))
    reasons.append(f"{c.travel_min} min walk")

    # Quality: the rating, pulled towards an average until there are enough reviews.
    if venue.rating is not None:
        count = venue.review_count or 0
        adjusted = ((count * venue.rating + RATING_PRIOR_WEIGHT * RATING_PRIOR)
                    / (count + RATING_PRIOR_WEIGHT))
        quality = min(1.0, max(0.0, (adjusted - RATING_FLOOR) / (RATING_CEILING - RATING_FLOOR)))
        reasons.append(f"rated {venue.rating:g} from {count:,} reviews")
    else:
        quality = 0.5

    weights = dict(WEIGHTS)
    if request.raining:
        weights["proximity"] += RAIN_PROXIMITY_BONUS
        reasons.append("raining around then, so a short walk counts for more")
    total = (weights["preference"] * preference + weights["timing"] * timing
             + weights["proximity"] * proximity + weights["quality"] * quality)
    score = round(total / sum(weights.values()), 4)
    return replace(c, score=score, reasons=tuple(reasons))


def open_until_text(arrival: datetime, closes_at: datetime) -> str:
    """'open until 18:00', 'open until 02:00 next day' or 'open 24 hours'."""
    arrival, closes_at = arrival.astimezone(LONDON), closes_at.astimezone(LONDON)
    if closes_at - arrival >= timedelta(hours=24):
        return "open 24 hours"
    suffix = " next day" if closes_at.date() != arrival.date() else ""
    return f"open until {closes_at:%H:%M}{suffix}"


# ── 3. Backups ──────────────────────────────────────────────────────────────

_BUSY_RANK = {"quiet": 0, "moderate": 1, "busy": 2}


def _qualifies(angle: str, c: Candidate, pick: Candidate) -> bool:
    if angle == "closer":
        return c.travel_min < pick.travel_min
    if angle == "quieter":
        mine, theirs = _BUSY_RANK.get(c.busyness.level), _BUSY_RANK.get(pick.busyness.level)
        return mine is not None and theirs is not None and mine < theirs
    if angle == "cheaper":
        return (c.venue.price_level is not None and pick.venue.price_level is not None
                and c.venue.price_level < pick.venue.price_level)
    if angle == "higher_rated":
        return (c.venue.rating is not None and pick.venue.rating is not None
                and c.venue.rating > pick.venue.rating)
    if angle == "open_later":
        return c.closes_at >= pick.closes_at + timedelta(minutes=OPEN_LATER_MIN)
    return angle == "next_best"


def _near_copy(c: Candidate, chosen: list[Candidate]) -> bool:
    return any(same_name(c.venue.name, other.venue.name)
               or distance_m((c.venue.lat, c.venue.lng), (other.venue.lat, other.venue.lng))
               < NEAR_COPY_METRES
               for other in chosen)


def backups(ranked: list[Candidate]) -> list[tuple[str, Candidate]]:
    """Up to four alternatives to ranked[0], each with a different angle.

    For each angle in turn, the best-scoring candidate that offers it and is
    not a near-copy of the pick or of a backup already chosen. An angle with
    no such candidate is skipped; `next_best` fills in last.
    """
    if not ranked:
        return []
    pick = ranked[0]
    chosen = [pick]
    out: list[tuple[str, Candidate]] = []
    for angle in ANGLES:
        if len(out) == MAX_BACKUPS:
            break
        for c in ranked[1:]:
            if _qualifies(angle, c, pick) and not _near_copy(c, chosen):
                chosen.append(c)
                out.append((angle, c))
                break
    return out
