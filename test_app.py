"""London Right Now, called the way Pippa calls it, with no network.

`LocalRunner` is the same code the production runtime uses, so a `Result`
that breaks the manifest fails here exactly as it would live. Every request
goes to `Web`, a fake that answers each data source from the seed venues
below, so the expected answers never change overnight.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from pippa_agentapp_sdk import (
    LocalRunner,
    OutputShapeViolation,
    RateLimitedError,
    Result,
    ServiceUnavailableError,
)

import scoring
from app import OUTCOME, App, BadTimeWindow, OutsideLondon, PlaceNotFound

#: Wednesday 18 November 2026, 17:40. London is on GMT in November.
NOW = datetime(2026, 11, 18, 17, 40, tzinfo=timezone.utc)
ORIGIN = (51.5194, -0.1270)
HERE = "51.5194,-0.1270"

GOOGLE_KEY = "fake-google-key-for-tests"  # pragma: allowlist secret
BESTTIME_KEY = "fake-besttime-key-for-tests"  # pragma: allowlist secret


# ── Seed data: OpenStreetMap venues around the origin ───────────────────────


def element(oid: int, name: str, kind: str, hours: str, dlat: float, *, tags=None, way=False) -> dict:
    key, value = kind.split("=")
    lat, lng = round(ORIGIN[0] + dlat, 6), ORIGIN[1]
    out = {"type": "way" if way else "node", "id": oid,
           "tags": {key: value, "name": name, "opening_hours": hours,
                    "addr:housenumber": str(oid), "addr:street": "Gower Street", **(tags or {})}}
    if way:
        out["center"] = {"lat": lat, "lon": lng}
    else:
        out["lat"], out["lon"] = lat, lng
    return out


SEED = [
    element(1, "Early Museum", "tourism=museum", "Mo-Su 10:00-18:00", 0.0005),            # 1 min
    element(2, "Late Museum", "tourism=museum", "Mo-Su 10:00-18:00; We 10:00-21:00", 0.004),  # 8 min
    element(3, "Corner Cafe", "amenity=cafe", "Mo-Su 07:00-19:00", 0.001,                  # 2 min
            tags={"internet_access": "wlan"}),
    element(4, "Night Owl Cafe", "amenity=cafe", "Mo-Su 08:00-23:00", 0.003),             # 6 min
    element(5, "Page Turners", "shop=books", "Mo-Sa 09:00-20:00; Su 11:00-17:00", -0.002),  # 4 min
    element(6, "Pasta Place", "amenity=restaurant", "Mo-Su 12:00-22:30", -0.003,           # 6 min
            tags={"cuisine": "italian"}),
    element(7, "Ramen Bar", "amenity=restaurant", "Mo-Su 11:30-15:00,17:00-23:00", -0.005,  # 10 min
            tags={"cuisine": "ramen;japanese"}),
    element(8, "Studio Gallery", "tourism=gallery", "Tu-Su 10:00-18:00", 0.002),          # 4 min
    element(9, "Riverside Gallery", "tourism=gallery", "Mo-Su 10:00-17:30", -0.004),      # 8 min
    element(10, "Far Cafe", "amenity=cafe", "24/7", 0.02, way=True),                      # 37 min
    element(11, "Mystery Cafe", "amenity=cafe", "sunrise-sunset", 0.0015),                 # unreadable
]


def as_json(value) -> bytes:
    return json.dumps(value).encode("utf-8")


def http(code: int, retry_after: int | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return urllib.error.HTTPError("https://example.invalid/", code, "error", headers, None)


def weather(rain: bool = False):
    def answer(url: str) -> bytes:
        day = parse_qs(urlsplit(url).query)["start_date"][0]
        return as_json({"hourly": {
            "time": [f"{day}T{h:02d}:00" for h in range(24)],
            "precipitation": [1.5 if rain else 0.0] * 24,
            "precipitation_probability": [90 if rain else 5] * 24,
        }})
    return answer


def besttime(by_name: dict, *, live: bool = False, day_int: int = 3):
    """Forecast: {name: {hour: intensity}}. Live: {name: percent}. Others: not found."""
    def answer(url: str) -> bytes:
        parts = urlsplit(url)
        assert parts.path.endswith("/live") == live, "asked for the wrong kind of busyness"
        value = by_name.get(parse_qs(parts.query)["venue_name"][0])
        if value is None:
            return as_json({"status": "Error", "message": "Could not find venue"})
        if live:
            return as_json({"status": "OK", "analysis": {
                "venue_live_busyness_available": True, "venue_live_busyness": value}})
        return as_json({"status": "OK", "analysis": [{
            "day_info": {"day_int": day_int},
            "hour_analysis": [{"hour": h, "intensity_txt": t} for h, t in value.items()],
        }]})
    return answer


HOSTS = {
    "overpass-api.de": "overpass", "overpass.kumi.systems": "mirror",
    "api.open-meteo.com": "weather", "api.postcodes.io": "postcodes",
    "nominatim.openstreetmap.org": "nominatim", "places.googleapis.com": "google",
    "besttime.app": "besttime",
}


class Web:
    """Stands in for `App._fetch`. Each answer is bytes, an exception, or a function of the URL."""

    def __init__(self, **answers) -> None:
        self.answers = {"overpass": as_json({"elements": SEED}), "weather": weather(), **answers}
        self.calls: list[tuple[str, str, dict]] = []

    def __call__(self, url: str, timeout: float = 6.0, **kwargs) -> bytes:
        source = HOSTS[urlsplit(url).hostname]
        self.calls.append((source, url, kwargs))
        if source == "mirror" and "mirror" not in self.answers:
            source = "overpass"       # the mirror says what the main server says
        if source not in self.answers:
            raise AssertionError(f"the app asked {source}, which this test did not expect")
        answer = self.answers[source]
        if isinstance(answer, BaseException):
            raise answer
        return answer(url) if callable(answer) else answer

    def asked(self) -> list[str]:
        return [source for source, _, _ in self.calls]


def ask(inputs: dict, web: Web | None = None, *, explicit: bool = False) -> Result:
    """Call the app with exactly these inputs."""
    app = App()
    app._fetch = web if web is not None else Web()
    return LocalRunner(app).call(OUTCOME, inputs, granted_secrets={}, now=NOW,
                                 request_id="test-1", explicit=explicit).result


def run(inputs: dict, web: Web | None = None, *, explicit: bool = False) -> Result:
    """As if the person also said they do not mind what kind of place, unless the
    test says. With nothing at all about what they like, the app asks instead."""
    return ask({"category": "any", **inputs}, web, explicit=explicit)


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    """Each test starts with neither key, as the app does until the owner adds them."""
    monkeypatch.delenv("GOOGLE_PLACES_API_KEY", raising=False)
    monkeypatch.delenv("BESTTIME_API_KEY", raising=False)


# ── The manifest ────────────────────────────────────────────────────────────


def test_the_manifest_is_valid():
    app = App()
    assert app.manifest.app_id == "hack.app_b0e7"
    assert [o.id for o in app.manifest.outcomes] == [OUTCOME]


def test_the_runner_enforces_the_manifest(monkeypatch):
    monkeypatch.setattr(App, "handle", lambda self, c: Result(data={"surprise": "hello"}))
    with pytest.raises(OutputShapeViolation):
        run({"location": HERE})


# ── Acceptance ──────────────────────────────────────────────────────────────


def test_a_museum_closing_before_the_end_is_not_offered():
    """Acceptance 1: the nearest museum shuts at 18:00, inside the two hours."""
    data = run({"location": HERE, "category": "museum"}).data
    assert data["status"] == "ok"
    assert data["pick_name"] == "Late Museum"
    assert data["pick_open_until"] == "21:00"
    assert data["pick_travel_min"] == 8
    assert data["pick_time_available_min"] == 112
    assert not any("Early Museum" in line for line in data["backups"])
    assert "duration_min: 120 (default for museum)" in data["assumptions"]


def test_a_gallery_tomorrow_afternoon_uses_the_forecast(monkeypatch):
    """Acceptance 3: 14:00 to 16:00 on Thursday, with BestTime's forecast."""
    monkeypatch.setenv("BESTTIME_API_KEY", BESTTIME_KEY)
    web = Web(besttime=besttime({
        "Studio Gallery": {14: "Low", 15: "High"},
        "Riverside Gallery": {14: "Average", 15: "Average"},
    }))
    result = run({"location": HERE, "category": "gallery",
                  "start": "2026-11-19T14:00:00+00:00", "end": "2026-11-19T16:00:00+00:00"}, web)
    data = result.data
    assert data["pick_name"] == "Studio Gallery"
    assert data["pick_busyness"] == "quiet"
    assert data["pick_busyness_source"] == "forecast"
    assert data["pick_busier_after"] == "15:00"
    assert data["pick_open_until"] == "18:00"
    assert data["pick_time_available_min"] == 116
    assert "quiet around 14:04 (forecast), busier after 15:00" in data["pick_reasons"]
    assert data["backups"] == [
        "next_best: Riverside Gallery (gallery), 8 min walk, open until 17:30, "
        "busyness moderate, id osm:node/9",
    ]
    assert "BestTime" in data["sources"]
    # The key went to BestTime and nowhere else, and is in nothing returned.
    assert all(source == "besttime" for source, url, _ in web.calls if BESTTIME_KEY in url)
    assert BESTTIME_KEY not in json.dumps(data) + (result.narrative or "")


def test_without_busyness_data_it_says_unknown():
    """Acceptance 4: no BestTime key, so busyness is never guessed."""
    data = run({"location": HERE}).data
    assert data["pick_busyness"] == "unknown"
    assert data["pick_busyness_source"] == "none"
    assert "pick_busier_after" not in data
    assert "busyness: unavailable (BestTime is not set up)" in data["assumptions"]
    assert all(line.split(", busyness ")[1].startswith("unknown") for line in data["backups"])


def test_backups_each_have_their_own_angle():
    """Acceptance 5: distinct angles, none of them the pick."""
    data = run({"location": HERE}).data
    assert data["pick_name"] == "Page Turners"
    angles = [line.split(":")[0] for line in data["backups"]]
    assert angles == ["closer", "open_later", "next_best"]
    assert set(angles) <= set(scoring.ANGLES)
    assert [line.split(": ")[1].split(" (")[0] for line in data["backups"]] == [
        "Corner Cafe", "Night Owl Cafe", "Pasta Place"]
    assert not any(line.endswith(f"id {data['pick_venue_id']}") for line in data["backups"])


@pytest.mark.parametrize("location", ["", "near me", "London"])
def test_no_place_asks_where(location):
    """Acceptance 7: with nowhere to start from, it asks instead of guessing,
    and asks the mood question in the same breath."""
    web = Web()
    inputs = {"category": "museum", "mood": ""}
    if location:
        inputs["location"] = location
    result = ask(inputs, web)
    assert result.data["status"] == "needs_input"
    assert result.data["needs"] == ["location", "mood"]
    assert result.data["questions"][0] == (
        "Where in London will you be: an area, a station or a postcode?")
    assert "pick_name" not in result.data
    assert web.calls == []


def test_turned_down_places_are_never_offered():
    """Acceptance 8: by name and by id."""
    first = run({"location": HERE}).data
    again = run({"location": HERE, "exclude_venues": [first["pick_name"]]}).data
    assert again["pick_name"] != first["pick_name"]
    assert not any(first["pick_name"] in line for line in again["backups"])
    by_id = run({"location": HERE, "exclude_venues": [first["pick_venue_id"]]}).data
    assert by_id["pick_venue_id"] != first["pick_venue_id"]


def test_the_same_question_gets_the_same_answer():
    assert run({"location": HERE}).data == run({"location": HERE}).data


# ── Asking about what the person likes ──────────────────────────────────────


def test_knowing_nothing_it_asks_everything_at_once():
    web = Web()
    result = ask({}, web)
    data = result.data
    assert data["status"] == "needs_input"
    assert data["needs"] == ["location", "activity", "mood"]
    assert len(data["questions"]) == 3
    assert data["questions"][2].startswith("Are you in the mood to explore somewhere new")
    assert result.narrative.startswith("Before I pick somewhere: Where in London")
    assert (data["assumptions"], data["preferences"], data["backups"]) == ([], [], [])
    assert web.calls == []


def test_a_day_without_a_time_asks_when():
    """"Sunday for 60 minutes near Bloomsbury": which part of Sunday?"""
    web = Web()
    data = ask({"location": HERE, "day": "Sunday", "duration_min": 60}, web).data
    assert data["needs"] == ["start_time", "activity", "mood"]
    assert data["questions"][0] == "What time on Sunday does your free time start?"
    assert web.calls == []


def test_a_day_with_a_time_does_not_ask():
    data = ask({"location": HERE, "category": "museum", "day": "Sunday",
                "start": "2026-11-22T14:00:00+00:00", "duration_min": 60}).data
    assert data["status"] == "ok"
    assert "free time: Sunday 14:00 to 15:00" in data["preferences"]


def test_saying_what_they_like_is_enough_to_answer():
    """A kind of place, a food or a feel: any one of them, and no question."""
    for said in ({"category": "cafe"}, {"cuisine": "ramen"}, {"vibe": ["quiet"]},
                 {"regulars": ["Corner Cafe"]}):
        data = ask({"location": HERE, **said}).data
        assert data["status"] == "ok", said
        assert data["needs"] == [] and data["questions"] == []


def test_answering_the_questions_gives_a_pick():
    """The follow-up: everything from the first message, plus the answers."""
    data = ask({"location": HERE, "day": "Sunday", "start": "2026-11-22T14:00:00+00:00",
                "duration_min": 60, "category": "museum", "mood": "explore"}).data
    assert data["status"] == "ok"
    assert data["preferences"] == ["kind of place: museum", "mood: explore",
                                   "free time: Sunday 14:00 to 15:00"]
    assert "somewhere different: not a known chain or one of your regulars" in data["pick_reasons"]


def test_in_the_mood_for_a_regular_without_naming_one_it_asks_which():
    data = ask({"mood": "familiar"}).data
    assert data["needs"] == ["location", "activity", "regulars"]
    assert data["questions"][-1] == "Which places do you usually go to?"


def test_a_regular_comes_first_when_that_is_the_mood():
    data = run({"location": HERE, "category": "cafe", "mood": "my usual",
                "regulars": ["Night Owl"]}).data
    assert data["pick_name"] == "Night Owl Cafe"
    assert "one of your regulars" in data["pick_reasons"]
    assert data["preferences"] == ["kind of place: cafe", "mood: familiar",
                                   "regular places: Night Owl"]


def test_exploring_passes_over_a_regular():
    usual = run({"location": HERE, "category": "cafe", "regulars": ["Corner Cafe"]}).data
    assert usual["pick_name"] == "Corner Cafe"
    new = run({"location": HERE, "category": "cafe", "mood": "somewhere new",
               "regulars": ["Corner Cafe"]}).data
    assert new["pick_name"] == "Night Owl Cafe"


def test_familiar_with_no_regulars_favours_a_chain():
    seed = [dict(e, tags={**e["tags"], "brand": "Night Owl Coffee"}) if e["id"] == 4 else e
            for e in SEED]
    data = run({"location": HERE, "category": "cafe", "mood": "familiar"},
               Web(overpass=as_json({"elements": seed}))).data
    assert data["pick_name"] == "Night Owl Cafe"
    assert "a chain you will know" in data["pick_reasons"]
    assert "regulars: none named, so well-known chains were favoured" in data["assumptions"]


def test_a_regular_that_is_not_open_is_reported():
    data = run({"location": HERE, "category": "cafe", "regulars": ["Prufrock"]}).data
    assert data["status"] == "ok"
    assert ("regulars: none of them is open for the whole window within 18 min walk"
            in data["assumptions"])


def test_a_mood_it_cannot_read_is_treated_as_either():
    data = run({"location": HERE, "mood": "sleepy"}).data
    assert data["status"] == "ok"
    assert "mood 'sleepy' not understood, so treated as either" in data["assumptions"]


def test_now_is_not_a_day_that_needs_a_time():
    assert ask({"location": HERE, "category": "cafe", "day": "now"}).data["status"] == "ok"


# ── Busyness ────────────────────────────────────────────────────────────────


def test_a_window_starting_now_uses_live_busyness(monkeypatch):
    monkeypatch.setenv("BESTTIME_API_KEY", BESTTIME_KEY)
    web = Web(besttime=besttime({"Corner Cafe": 85, "Night Owl Cafe": 20}, live=True))
    data = run({"location": HERE, "category": "cafe"}, web).data
    assert data["pick_name"] == "Night Owl Cafe"
    assert (data["pick_busyness"], data["pick_busyness_source"]) == ("quiet", "live")
    assert "quiet now" in data["pick_reasons"]
    assert data["backups"] == [
        "closer: Corner Cafe (cafe), 2 min walk, open until 19:00, busyness busy, id osm:node/3"]


def test_a_venue_besttime_does_not_know_stays_unknown(monkeypatch):
    monkeypatch.setenv("BESTTIME_API_KEY", BESTTIME_KEY)
    web = Web(besttime=besttime({"Studio Gallery": {14: "Low", 15: "Low"}}))
    data = run({"location": HERE, "category": "gallery",
                "start": "2026-11-19T14:00:00+00:00", "end": "2026-11-19T16:00:00+00:00"}, web).data
    assert "busyness: unavailable for 1 of the 2 best places" in data["assumptions"]
    assert any(line.startswith("next_best: Riverside Gallery") and "busyness unknown" in line
               for line in data["backups"])


def test_besttime_failing_is_not_fatal(monkeypatch):
    monkeypatch.setenv("BESTTIME_API_KEY", BESTTIME_KEY)
    data = run({"location": HERE}, Web(besttime=http(401))).data
    assert data["status"] == "ok"
    assert data["pick_busyness"] == "unknown"
    assert "BestTime" not in data["sources"]


# ── Preferences ─────────────────────────────────────────────────────────────


def test_a_vibe_is_matched_to_what_the_map_says():
    data = run({"location": HERE, "category": "cafe", "vibe": ["wifi"]}).data
    assert data["pick_name"] == "Corner Cafe"
    assert "matches: wifi" in data["pick_reasons"]


def test_a_vibe_with_no_data_is_reported():
    data = run({"location": HERE, "vibe": ["romantic"]}).data
    assert "vibe 'romantic': no data to match it on" in data["assumptions"]


def test_a_cuisine_finds_the_restaurant():
    data = run({"location": HERE, "category": "restaurant", "cuisine": "ramen"}).data
    assert data["pick_name"] == "Ramen Bar"


def test_an_uncovered_category_considers_everything():
    data = run({"location": HERE, "category": "pub"}).data
    assert data["status"] == "ok"
    assert any(a.startswith("category 'pub' is not covered") for a in data["assumptions"])


# ── The time window ─────────────────────────────────────────────────────────


def test_a_duration_sets_the_end_and_the_walking_limit():
    data = run({"location": HERE, "duration_min": 30}).data
    assert not any(a.startswith("duration_min:") for a in data["assumptions"])
    assert "travel: walking time estimated from distance, at most 12 min" in data["assumptions"]


def test_an_end_before_the_start_is_refused():
    with pytest.raises(BadTimeWindow):
        run({"location": HERE, "start": "2026-11-19T15:00:00+00:00", "end": "2026-11-19T14:00:00+00:00"})


def test_nothing_open_says_so():
    data = run({"location": HERE, "category": "bookshop", "start": "2026-11-18T21:00:00+00:00"}).data
    assert data["status"] == "no_results"
    assert "pick_name" not in data
    assert data["backups"] == []


# ── Where ───────────────────────────────────────────────────────────────────


def test_a_place_name_is_looked_up():
    web = Web(nominatim=as_json([{"lat": "51.5194", "lon": "-0.1270",
                                  "display_name": "Bloomsbury, London Borough of Camden"}]))
    result = run({"location": "Bloomsbury"}, web)
    assert result.data["sources"][0] == "OpenStreetMap Nominatim"
    assert "location: 'Bloomsbury' taken as Bloomsbury, London Borough of Camden" in result.data["assumptions"]
    assert "from Bloomsbury" in result.narrative


def test_a_postcode_is_looked_up():
    web = Web(postcodes=as_json({"status": 200, "result": {"latitude": 51.5194, "longitude": -0.127}}))
    data = run({"location": "WC1E 7HU"}, web).data
    assert data["sources"][0] == "postcodes.io"
    assert data["status"] == "ok"


def test_an_unknown_postcode_says_what_to_type():
    with pytest.raises(PlaceNotFound):
        run({"location": "ZZ9 9ZZ"}, Web(postcodes=http(404)))


def test_a_place_that_cannot_be_found_says_what_to_type():
    with pytest.raises(PlaceNotFound) as failure:
        run({"location": "Narnia"}, Web(nominatim=as_json([])))
    assert failure.value.user_message == (
        "I could not find Narnia in London. Try an area, a station or a postcode.")


def test_outside_london_is_refused():
    web = Web()
    with pytest.raises(OutsideLondon) as failure:
        run({"location": "53.4808,-2.2426"}, web)
    assert "outside London" in failure.value.user_message
    assert web.calls == []


# ── When a source fails ─────────────────────────────────────────────────────


def test_the_overpass_mirror_is_tried_when_the_main_server_fails():
    web = Web(overpass=http(504), mirror=as_json({"elements": SEED}))
    assert run({"location": HERE}, web).data["status"] == "ok"
    assert web.asked().count("overpass") == 1 and web.asked().count("mirror") == 1


def test_an_overpass_timeout_inside_a_200_is_a_failure():
    web = Web(overpass=as_json({"remark": "runtime error: Query timed out", "elements": []}),
              mirror=as_json({"elements": SEED}))
    assert run({"location": HERE}, web).data["status"] == "ok"
    assert "mirror" in web.asked()


def test_no_venue_source_is_a_retryable_error():
    with pytest.raises(ServiceUnavailableError):
        run({"location": HERE}, Web(overpass=http(503)))
    with pytest.raises(ServiceUnavailableError):
        run({"location": HERE}, Web(overpass=TimeoutError("slow")))


def test_rate_limited_says_try_again_shortly():
    with pytest.raises(RateLimitedError):
        run({"location": HERE}, Web(overpass=http(429, retry_after=30)))


def test_no_weather_is_reported_not_fatal():
    data = run({"location": HERE}, Web(weather=TimeoutError("slow"))).data
    assert data["status"] == "ok"
    assert "weather: unavailable" in data["assumptions"]
    assert "Open-Meteo" not in data["sources"]


def test_rain_is_taken_into_account():
    data = run({"location": HERE}, Web(weather=weather(rain=True))).data
    assert "weather: rain expected around 17:00" in data["assumptions"]
    assert "raining around then, so a short walk counts for more" in data["pick_reasons"]


# ── Google Places, when its key is set ──────────────────────────────────────

GOOGLE_PLACES = {"places": [{
    "id": "ChIJtest", "displayName": {"text": "Google Cafe"},
    "location": {"latitude": ORIGIN[0] + 0.001, "longitude": ORIGIN[1]},
    "primaryType": "cafe", "types": ["cafe", "coffee_shop"],
    "rating": 4.6, "userRatingCount": 1200, "priceLevel": "PRICE_LEVEL_INEXPENSIVE",
    "regularOpeningHours": {"periods": [{"open": {"day": 0, "hour": 0, "minute": 0}}]},
    "googleMapsUri": "https://maps.google.com/?cid=1",
    "formattedAddress": "1 Gower St, London WC1E 6HA, UK",
}]}


def test_google_places_is_used_when_its_key_is_set(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", GOOGLE_KEY)
    web = Web(google=as_json(GOOGLE_PLACES))
    result = run({"location": HERE, "category": "cafe"}, web)
    data = result.data
    assert data["pick_name"] == "Google Cafe"
    assert (data["pick_rating"], data["pick_review_count"], data["pick_price_level"]) == (4.6, 1200, 1)
    assert data["pick_open_until"] == "open 24 hours"
    assert data["pick_maps_url"] == "https://maps.google.com/?cid=1"
    assert data["sources"] == ["Google Places", "Open-Meteo"]
    assert "overpass" not in web.asked()
    # The key travels only in the unredirected header, never in the URL or body.
    (_, url, kwargs), = [call for call in web.calls if call[0] == "google"]
    assert kwargs["secret_header"] == ("X-Goog-Api-Key", GOOGLE_KEY)
    assert GOOGLE_KEY not in url
    assert GOOGLE_KEY not in json.dumps(kwargs.get("headers") or {})
    assert GOOGLE_KEY not in json.dumps(data) + (result.narrative or "")


def test_google_failing_falls_back_to_openstreetmap(monkeypatch):
    monkeypatch.setenv("GOOGLE_PLACES_API_KEY", GOOGLE_KEY)
    data = run({"location": HERE}, Web(google=http(403))).data
    assert data["status"] == "ok"
    assert "OpenStreetMap" in data["sources"]
    assert "venues: Google Places unavailable, so OpenStreetMap was used" in data["assumptions"]


# ── Picked by name ──────────────────────────────────────────────────────────


def test_picked_by_name_it_answers_and_asks_as_usual():
    """`@London Right Now ...`: the same answer, and the same question when there is no place."""
    assert run({"location": HERE}, explicit=True).data == run({"location": HERE}).data
    assert ask({}, explicit=True).data["needs"] == ask({}).data["needs"]
