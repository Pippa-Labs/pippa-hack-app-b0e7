"""The data sources: how to ask each one, and how to read what comes back.

Nothing here does HTTP. Each adapter builds a request (a URL, and a body for
the POSTs) and parses a response it is handed; the app's single `_fetch`
method does the network, so tests replace that one method and nothing else.

Every parser checks for emptiness before it indexes. The commonest failure
of a public API is not an error but a `200` that contains nothing.

| Source          | Used for                               | Key            |
|-----------------|----------------------------------------|----------------|
| postcodes.io    | a postcode or outward code -> lat/lng  | none           |
| Nominatim (OSM) | a place or station name -> lat/lng     | none           |
| Overpass (OSM)  | venues and opening hours               | none           |
| Open-Meteo      | is it raining around arrival           | none           |
| Google Places   | venues with ratings, prices, hours     | optional       |
| BestTime        | live or forecast busyness              | optional       |
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from urllib.parse import quote, urlencode

from hours import LONDON, from_google_periods, parse_osm
from scoring import UNKNOWN, Busyness, Venue


class BadResponse(Exception):
    """A response that could not be read. Internal: the app turns it into its own error."""


def _json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BadResponse(f"not JSON: {exc}") from None


def maps_url(lat: float, lng: float) -> str:
    return "https://www.google.com/maps/search/?" + urlencode(
        {"api": "1", "query": f"{lat:.6f},{lng:.6f}"})


# ── Where the person is ─────────────────────────────────────────────────────

#: Greater London, roughly. A place outside it is refused, not answered.
LONDON_BOX = {"south": 51.28, "west": -0.52, "north": 51.70, "east": 0.34}

COORDINATES = re.compile(r"^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$")
POSTCODE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.IGNORECASE)
OUTCODE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?$", re.IGNORECASE)


def in_london(lat: float, lng: float) -> bool:
    return (LONDON_BOX["south"] <= lat <= LONDON_BOX["north"]
            and LONDON_BOX["west"] <= lng <= LONDON_BOX["east"])


def postcodes_url(text: str) -> str:
    code = text.strip().upper()
    kind = "postcodes" if POSTCODE.match(code) else "outcodes"
    return f"https://api.postcodes.io/{kind}/{quote(code, safe='')}"


def parse_postcodes(raw: bytes) -> tuple[float, float] | None:
    result = (_json(raw) or {}).get("result")
    if not isinstance(result, dict):
        return None
    lat, lng = result.get("latitude"), result.get("longitude")
    if lat is None or lng is None:
        return None
    return float(lat), float(lng)


def nominatim_url(text: str) -> str:
    box = LONDON_BOX
    return "https://nominatim.openstreetmap.org/search?" + urlencode({
        "q": text, "format": "jsonv2", "limit": "1", "countrycodes": "gb",
        # Bounded to London, so "Victoria" is the station, not the state.
        "viewbox": f"{box['west']},{box['north']},{box['east']},{box['south']}",
        "bounded": "1",
    })


def parse_nominatim(raw: bytes) -> tuple[float, float, str] | None:
    results = _json(raw)
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    try:
        return float(first["lat"]), float(first["lon"]), str(first.get("display_name", ""))
    except (KeyError, TypeError, ValueError):
        return None


# ── Venues: OpenStreetMap through Overpass ──────────────────────────────────

OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",   # a mirror, tried second
)

OSM_FILTERS = {
    "cafe": '["amenity"="cafe"]',
    "restaurant": '["amenity"="restaurant"]',
    "museum": '["tourism"="museum"]',
    "gallery": '["tourism"="gallery"]',
    "bookshop": '["shop"="books"]',
}


def overpass_body(categories, origin: tuple[float, float], radius_m: int, timeout_s: int) -> bytes:
    """Named places of these kinds with published opening hours, within the radius."""
    lat, lng = float(origin[0]), float(origin[1])
    around = f"(around:{int(radius_m)},{lat:.6f},{lng:.6f})"
    parts = "".join(f'nwr{OSM_FILTERS[c]}["name"]["opening_hours"]{around};' for c in categories)
    query = f"[out:json][timeout:{int(timeout_s)}];({parts});out center tags;"
    return urlencode({"data": query}).encode("ascii")


def _osm_category(tags: dict) -> str | None:
    if tags.get("amenity") == "cafe":
        return "cafe"
    if tags.get("amenity") == "restaurant":
        return "restaurant"
    if tags.get("tourism") == "museum":
        return "museum"
    if tags.get("tourism") == "gallery":
        return "gallery"
    if tags.get("shop") == "books":
        return "bookshop"
    return None


def _osm_features(tags: dict) -> tuple[frozenset[str], frozenset[str]]:
    has: set[str] = set()
    lacks: set[str] = set()

    def read(feature: str, key: str, yes=("yes", "only", "designated"), no=("no",)):
        value = str(tags.get(key, "")).lower()
        if value in yes:
            has.add(feature)
        elif value in no:
            lacks.add(feature)

    read("wifi", "internet_access", yes=("wlan", "yes", "wifi", "free"))
    read("outdoor_seating", "outdoor_seating")
    read("wheelchair", "wheelchair")          # "limited" is neither
    read("vegan", "diet:vegan")
    read("vegetarian", "diet:vegetarian")
    read("gluten_free", "diet:gluten_free")
    read("dogs", "dog", yes=("yes", "leashed"))
    read("children", "kids_area")
    return frozenset(has), frozenset(lacks)


def parse_overpass(raw: bytes) -> tuple[list[Venue], int]:
    """The venues, and how many were left out because their hours could not be read."""
    data = _json(raw)
    if not isinstance(data, dict):
        raise BadResponse("overpass answered with something other than an object")
    remark = str(data.get("remark", ""))
    if "runtime error" in remark.lower() or "timed out" in remark.lower():
        # Overpass reports its own timeouts inside a 200.
        raise BadResponse("overpass ran out of time")
    venues: list[Venue] = []
    unreadable = 0
    for element in data.get("elements") or []:
        tags = element.get("tags") or {}
        category = _osm_category(tags)
        name = str(tags.get("name", "")).strip()
        lat = element.get("lat", (element.get("center") or {}).get("lat"))
        lng = element.get("lon", (element.get("center") or {}).get("lon"))
        if not category or not name or lat is None or lng is None:
            continue
        hours = parse_osm(tags.get("opening_hours"))
        if hours is None:
            unreadable += 1
            continue
        street = " ".join(p for p in (tags.get("addr:housenumber"), tags.get("addr:street")) if p)
        address = ", ".join(p for p in (street, tags.get("addr:postcode")) if p) or None
        cuisines = tuple(c.strip().lower() for c in str(tags.get("cuisine", "")).split(";") if c.strip())
        has, lacks = _osm_features(tags)
        description = str(tags.get("description", "")).strip()
        venues.append(Venue(
            venue_id=f"osm:{element.get('type', 'node')}/{element.get('id')}",
            name=name, category=category, lat=float(lat), lng=float(lng), hours=hours,
            maps_url=maps_url(float(lat), float(lng)), address=address,
            cuisines=cuisines, has=has, lacks=lacks,
            tip=description[:200] or None,
        ))
    return venues, unreadable


# ── Venues: Google Places (New), when its key is set ────────────────────────

GOOGLE_URL = "https://places.googleapis.com/v1/places:searchNearby"
GOOGLE_TYPES = {
    "cafe": ("cafe", "coffee_shop"),
    "museum": ("museum",),
    "gallery": ("art_gallery",),
    "bookshop": ("book_store",),
    "restaurant": ("restaurant",),
}
GOOGLE_FIELDS = ",".join(f"places.{field}" for field in (
    "id", "displayName", "formattedAddress", "location", "rating", "userRatingCount",
    "priceLevel", "regularOpeningHours", "googleMapsUri", "primaryType", "types",
    "editorialSummary", "outdoorSeating", "allowsDogs", "goodForChildren",
    "servesVegetarianFood", "accessibilityOptions",
))
GOOGLE_PRICE = {
    "PRICE_LEVEL_FREE": 0, "PRICE_LEVEL_INEXPENSIVE": 1, "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3, "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}


def google_body(categories, origin: tuple[float, float], radius_m: int) -> bytes:
    types = [t for c in categories for t in GOOGLE_TYPES[c]]
    return json.dumps({
        "includedTypes": types,
        "maxResultCount": 20,
        "rankPreference": "POPULARITY",
        "locationRestriction": {"circle": {
            "center": {"latitude": float(origin[0]), "longitude": float(origin[1])},
            "radius": float(max(50, min(radius_m, 50_000))),
        }},
    }).encode("utf-8")


def _google_category(place: dict) -> str | None:
    primary = str(place.get("primaryType", ""))
    types = [str(t) for t in place.get("types") or []]
    for candidate in (primary, *types):
        for category, names in GOOGLE_TYPES.items():
            if candidate in names:
                return category
        if candidate.endswith("_restaurant"):
            return "restaurant"
    return None


def parse_google(raw: bytes) -> tuple[list[Venue], int]:
    data = _json(raw)
    if not isinstance(data, dict):
        raise BadResponse("google answered with something other than an object")
    venues: list[Venue] = []
    unreadable = 0
    for place in data.get("places") or []:     # no places at all: the key is missing
        category = _google_category(place)
        name = str((place.get("displayName") or {}).get("text", "")).strip()
        location = place.get("location") or {}
        lat, lng = location.get("latitude"), location.get("longitude")
        if not category or not name or lat is None or lng is None or not place.get("id"):
            continue
        hours = from_google_periods((place.get("regularOpeningHours") or {}).get("periods"))
        if hours is None:
            unreadable += 1
            continue
        has: set[str] = set()
        lacks: set[str] = set()
        for feature, value in (
            ("outdoor_seating", place.get("outdoorSeating")),
            ("dogs", place.get("allowsDogs")),
            ("children", place.get("goodForChildren")),
            ("vegetarian", place.get("servesVegetarianFood")),
            ("wheelchair", (place.get("accessibilityOptions") or {}).get("wheelchairAccessibleEntrance")),
        ):
            if value is True:
                has.add(feature)
            elif value is False:
                lacks.add(feature)
        types = [str(t) for t in place.get("types") or []]
        cuisines = tuple(t[: -len("_restaurant")] for t in types if t.endswith("_restaurant"))
        if "coffee_shop" in types:
            cuisines += ("coffee_shop",)
        rating = place.get("rating")
        count = place.get("userRatingCount")
        summary = str((place.get("editorialSummary") or {}).get("text", "")).strip()
        venues.append(Venue(
            venue_id=f"google:{place['id']}", name=name, category=category,
            lat=float(lat), lng=float(lng), hours=hours,
            maps_url=str(place.get("googleMapsUri") or maps_url(float(lat), float(lng))),
            address=str(place.get("formattedAddress") or "") or None,
            rating=float(rating) if rating is not None else None,
            review_count=int(count) if count is not None else None,
            price_level=GOOGLE_PRICE.get(str(place.get("priceLevel", ""))),
            cuisines=cuisines, has=frozenset(has), lacks=frozenset(lacks),
            tip=summary or None,
        ))
    return venues, unreadable


# ── Weather: Open-Meteo ─────────────────────────────────────────────────────

RAIN_MM = 0.2
RAIN_PROBABILITY = 60


def open_meteo_url(origin: tuple[float, float], when: datetime) -> str:
    day = when.astimezone(LONDON).date().isoformat()
    return "https://api.open-meteo.com/v1/forecast?" + urlencode({
        "latitude": f"{origin[0]:.4f}", "longitude": f"{origin[1]:.4f}",
        "hourly": "precipitation,precipitation_probability",
        "timezone": "Europe/London", "start_date": day, "end_date": day,
    })


def parse_rain(raw: bytes, when: datetime) -> bool | None:
    """Whether rain is expected in the hour of `when`; None when there is no figure."""
    hourly = (_json(raw) or {}).get("hourly") or {}
    times = hourly.get("time") or []
    key = when.astimezone(LONDON).strftime("%Y-%m-%dT%H:00")
    if key not in times:
        return None
    index = times.index(key)
    rain = (hourly.get("precipitation") or [None] * len(times))[index]
    chance = (hourly.get("precipitation_probability") or [None] * len(times))[index]
    if rain is None and chance is None:
        return None
    return (rain or 0) >= RAIN_MM or (chance or 0) >= RAIN_PROBABILITY


# ── Busyness: BestTime, when its key is set ─────────────────────────────────

BESTTIME_FORECAST_URL = "https://besttime.app/api/v1/forecasts"
BESTTIME_LIVE_URL = "https://besttime.app/api/v1/forecasts/live"

#: BestTime's five-point words, as this app's three levels. "Closed" is not a
#: level: it means BestTime has no reading, so the answer stays "unknown".
INTENSITY_LEVEL = {"low": "quiet", "below average": "quiet", "average": "moderate",
                   "above average": "busy", "high": "busy"}
#: A live percentage (of the venue's usual weekly peak) as a level.
LIVE_QUIET_BELOW, LIVE_BUSY_FROM = 40, 70
_RANK = {"quiet": 0, "moderate": 1, "busy": 2}


def besttime_url(live: bool, key: str, name: str, address: str) -> str:
    """BestTime takes its parameters in the query string of a POST."""
    base = BESTTIME_LIVE_URL if live else BESTTIME_FORECAST_URL
    return base + "?" + urlencode(
        {"api_key_private": key, "venue_name": name, "venue_address": address})


def _besttime_day(when: datetime) -> int:
    """BestTime's day: Monday is 0, and each day runs from 6am to 5am."""
    return when.weekday() if when.hour >= 6 else (when.weekday() - 1) % 7


def _hour_level(analysis: list, when: datetime) -> str | None:
    day = _besttime_day(when)
    for entry in analysis:
        if not isinstance(entry, dict) or (entry.get("day_info") or {}).get("day_int") != day:
            continue
        for hour in entry.get("hour_analysis") or []:
            if isinstance(hour, dict) and hour.get("hour") == when.hour:
                return INTENSITY_LEVEL.get(str(hour.get("intensity_txt", "")).strip().lower())
    return None


def parse_besttime_forecast(raw: bytes, arrival: datetime, end: datetime) -> Busyness:
    data = _json(raw)
    if not isinstance(data, dict) or str(data.get("status", "")).upper() != "OK":
        return UNKNOWN
    analysis = data.get("analysis")
    if not isinstance(analysis, list):
        return UNKNOWN
    arrival, end = arrival.astimezone(LONDON), end.astimezone(LONDON)
    level = _hour_level(analysis, arrival)
    if level is None:
        return UNKNOWN
    busier_after = None
    hour = arrival.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while hour < end:
        later = _hour_level(analysis, hour)
        if later is not None and _RANK[later] > _RANK[level]:
            busier_after = f"{hour:%H:%M}"
            break
        hour += timedelta(hours=1)
    return Busyness(level=level, source="forecast", busier_after=busier_after)


def _percent_level(value) -> str | None:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return None
    if percent < LIVE_QUIET_BELOW:
        return "quiet"
    return "moderate" if percent < LIVE_BUSY_FROM else "busy"


def parse_besttime_live(raw: bytes) -> Busyness:
    data = _json(raw)
    if not isinstance(data, dict) or str(data.get("status", "")).upper() != "OK":
        return UNKNOWN
    analysis = data.get("analysis") or {}
    if not isinstance(analysis, dict):
        return UNKNOWN
    if analysis.get("venue_live_busyness_available") is True:
        level = _percent_level(analysis.get("venue_live_busyness"))
        if level:
            return Busyness(level=level, source="live")
    if analysis.get("venue_forecast_busyness_available") is True:
        level = _percent_level(analysis.get("venue_forecasted_busyness"))
        if level:
            return Busyness(level=level, source="forecast")
    return UNKNOWN
