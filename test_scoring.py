"""The rules that choose the venue, tested on plain data with no network."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

import scoring
from hours import LONDON, parse_osm
from scoring import Busyness, Candidate, Request, Venue

ORIGIN = (51.5194, -0.1270)
#: Wednesday 18 November 2026, 17:40 in London.
START = datetime(2026, 11, 18, 17, 40, tzinfo=LONDON)


def venue(vid: str, name: str, category: str = "cafe", dlat: float = 0.001,
          hours: str = "Mo-Su 08:00-22:00", **extra) -> Venue:
    lat, lng = ORIGIN[0] + dlat, ORIGIN[1]
    return Venue(venue_id=vid, name=name, category=category, lat=lat, lng=lng,
                 hours=parse_osm(hours), maps_url="https://maps.example/x", **extra)


def request(category: str = "any", minutes: int = 60, **extra) -> Request:
    return Request(category=category, origin=ORIGIN, start=START,
                   end=START + timedelta(minutes=minutes), **extra)


def names(candidates) -> list[str]:
    return [c.venue.name for c in candidates]


# ── Travel ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("minutes, cap", [(60, 24), (120, 30), (45, 18), (5, 2), (1, 1)])
def test_travel_is_capped_at_a_share_of_the_window(minutes, cap):
    assert request(minutes=minutes).max_travel_min == cap


def test_walking_minutes_from_distance():
    # 111 m north, 1.3 times longer on real streets, at 80 m a minute.
    assert scoring.walk_minutes(ORIGIN, ORIGIN[0] + 0.001, ORIGIN[1]) == 2


def test_search_radius_matches_the_walk():
    assert scoring.search_radius_m(24) == 1476


# ── Hard filters ────────────────────────────────────────────────────────────


def test_a_place_closing_before_the_end_is_left_out():
    """Acceptance 1: the nearest museum shuts at 18:00, so it is not offered."""
    early = venue("m1", "Early Museum", "museum", 0.0005, "Mo-Su 10:00-18:00")
    late = venue("m2", "Late Museum", "museum", 0.004, "Mo-Su 10:00-18:00; We 10:00-21:00")
    kept = scoring.shortlist(request("museum", 120), [early, late])
    assert names(kept) == ["Late Museum"]
    assert kept[0].closes_at == datetime(2026, 11, 18, 21, 0, tzinfo=LONDON)


def test_wrong_kind_of_place_is_left_out():
    kept = scoring.shortlist(request("bookshop"), [venue("c", "Cafe"), venue("b", "Books", "bookshop")])
    assert names(kept) == ["Books"]


def test_turned_down_places_are_left_out_by_name_or_id():
    places = [venue("osm:1", "The Corner Cafe"), venue("osm:2", "Other Cafe", dlat=0.002),
              venue("osm:3", "Third Cafe", dlat=0.003)]
    kept = scoring.shortlist(request(exclude=("corner cafe", "osm:3")), places)
    assert names(kept) == ["Other Cafe"]


def test_a_stated_need_drops_places_known_not_to_meet_it():
    places = [venue("a", "Steps Cafe", lacks=frozenset({"wheelchair"})),
              venue("b", "Unknown Cafe", dlat=0.002),
              venue("c", "Ramp Cafe", dlat=0.003, has=frozenset({"wheelchair"}))]
    kept = scoring.shortlist(request(vibes=("wheelchair accessible",)), places)
    assert names(kept) == ["Unknown Cafe", "Ramp Cafe"]


def test_too_far_to_walk_is_left_out():
    kept = scoring.shortlist(request(), [venue("far", "Far Cafe", dlat=0.02, hours="24/7")])
    assert kept == []


def test_the_same_place_twice_counts_once():
    once = venue("x", "Cafe")
    assert len(scoring.shortlist(request(), [once, once])) == 1


def test_time_available_runs_from_arrival_to_the_end():
    kept = scoring.shortlist(request(minutes=60), [venue("a", "Cafe", dlat=0.001)])
    assert kept[0].travel_min == 2
    assert kept[0].time_available_min == 58


# ── The score ───────────────────────────────────────────────────────────────


def ranked(req: Request, places) -> list[Candidate]:
    return scoring.rank(req, scoring.shortlist(req, places))


def test_scores_stay_between_zero_and_one():
    places = [venue(str(i), f"Cafe {i}", dlat=0.001 * i, rating=3.0 + 0.4 * i, review_count=100 * i)
              for i in range(1, 6)]
    for candidate in ranked(request(cuisine="pasta", budget=2), places):
        assert 0.0 <= candidate.score <= 1.0


def test_the_same_inputs_give_the_same_order():
    places = [venue(str(i), f"Cafe {i}", dlat=0.0007 * i) for i in range(1, 8)]
    assert names(ranked(request(), places)) == names(ranked(request(), list(reversed(places))))


def test_a_tie_breaks_on_travel_then_id():
    twin_b = venue("x:2", "Twin B")
    twin_a = venue("x:1", "Twin A")
    assert names(ranked(request(), [twin_b, twin_a])) == ["Twin A", "Twin B"]


def test_few_reviews_do_not_beat_many():
    """A 5.0 from three reviews is pulled towards average; 4.6 from 3,000 is not."""
    lucky = venue("a", "Lucky", rating=5.0, review_count=3)
    proven = venue("b", "Proven", rating=4.6, review_count=3000)
    assert names(ranked(request(), [lucky, proven]))[0] == "Proven"


def test_a_cuisine_match_wins():
    plain = venue("a", "Plain", "restaurant", 0.001, cuisines=("burger",))
    italian = venue("b", "Italian", "restaurant", 0.003, cuisines=("italian",))
    top = ranked(request("restaurant", cuisine="pasta"), [plain, italian])[0]
    assert top.venue.name == "Italian"
    assert "matches: pasta" in top.reasons


def test_asking_for_quiet_prefers_a_quiet_place():
    req = request(vibes=("quiet",))
    candidates = scoring.shortlist(req, [venue("a", "Loud", dlat=0.001), venue("b", "Hush", dlat=0.003)])
    levels = {"Loud": Busyness("busy", "live"), "Hush": Busyness("quiet", "live")}
    candidates = [replace(c, busyness=levels[c.venue.name]) for c in candidates]
    top = scoring.rank(req, candidates)[0]
    assert top.venue.name == "Hush"
    assert "quiet now" in top.reasons


def test_over_budget_scores_lower():
    req = request("restaurant", budget=1)
    cheap = venue("a", "Cheap", "restaurant", 0.002, price_level=1)
    dear = venue("b", "Dear", "restaurant", 0.002, price_level=4)
    top, second = ranked(req, [dear, cheap])
    assert (top.venue.name, second.venue.name) == ("Cheap", "Dear")
    assert "within budget" in top.reasons and "above budget" in second.reasons


def test_rain_makes_a_short_walk_count_for_more():
    near = venue("a", "Near", dlat=0.0005)
    far = venue("b", "Far", dlat=0.005, rating=4.9, review_count=5000)

    def gap(raining: bool) -> float:
        by_name = {c.venue.name: c.score for c in ranked(request(raining=raining), [near, far])}
        return by_name["Near"] - by_name["Far"]

    assert gap(True) > gap(False)
    wet = ranked(request(raining=True), [near])[0]
    assert "raining around then, so a short walk counts for more" in wet.reasons


def test_reasons_say_when_it_closes_and_how_far():
    top = ranked(request(), [venue("a", "Cafe", dlat=0.001, hours="Mo-Su 08:00-19:00")])[0]
    assert "open until 19:00" in top.reasons
    assert "2 min walk" in top.reasons


def test_unknown_busyness_is_never_claimed():
    top = ranked(request(), [venue("a", "Cafe")])[0]
    assert top.busyness.level == "unknown"
    assert not any("quiet" in r or "busy" in r for r in top.reasons)


# ── Mood: somewhere new, or a regular ───────────────────────────────────────


def test_exploring_prefers_an_independent_to_a_chain():
    chain = venue("a", "Big Brand Coffee", dlat=0.001, chain=True)
    indie = venue("b", "Little Cafe", dlat=0.003)
    assert names(ranked(request(), [chain, indie]))[0] == "Big Brand Coffee"
    top = ranked(request(mood="explore"), [chain, indie])[0]
    assert top.venue.name == "Little Cafe"
    assert "somewhere different: not a known chain or one of your regulars" in top.reasons


def test_exploring_skips_a_regular():
    regular = venue("a", "Prufrock Coffee", dlat=0.001)
    other = venue("b", "Other Cafe", dlat=0.004)
    assert names(ranked(request(mood="explore", regulars=("Prufrock",)), [regular, other]))[0] == "Other Cafe"


def test_familiar_puts_a_regular_first():
    near = venue("a", "Near Cafe", dlat=0.001)
    usual = venue("b", "Prufrock Coffee", dlat=0.005)
    top = ranked(request(mood="familiar", regulars=("prufrock",)), [near, usual])[0]
    assert top.venue.name == "Prufrock Coffee"
    assert "one of your regulars" in top.reasons


def test_exploring_minds_the_walk_less():
    near, far = venue("a", "Near", dlat=0.0005), venue("b", "Far", dlat=0.005)

    def gap(mood: str) -> float:
        by_name = {c.venue.name: c.score for c in ranked(request(mood=mood), [near, far])}
        return by_name["Near"] - by_name["Far"]

    assert 0 < gap("explore") < gap("either")


def test_no_mood_and_no_regulars_leaves_the_score_alone():
    plain = ranked(request(), [venue("a", "Cafe")])[0]
    assert not any("regular" in r or "chain" in r for r in plain.reasons)


def test_a_regular_is_matched_on_whole_words():
    assert scoring.is_regular("Prufrock Coffee", ["Prufrock"])
    assert scoring.is_regular("The Corner Cafe", ["corner cafe"])
    assert not scoring.is_regular("Ruffles", ["Prufrock"])
    assert not scoring.is_regular("Cafe", ["caf"])


# ── Reading the request ─────────────────────────────────────────────────────


def test_vibes_become_features_or_are_reported_unmatched():
    assert scoring.wanted_features(["laptop-friendly", "romantic"]) == (["wifi"], ["romantic"])


def test_a_dish_means_its_cuisines():
    assert "italian" in scoring.cuisine_terms("pasta")


def test_names_match_without_the_and_punctuation():
    assert scoring.same_name("The Corner & Co.", "corner and co")
    assert not scoring.same_name("Corner Cafe", "Corner Kitchen")


# ── Closing-time wording ────────────────────────────────────────────────────


def test_open_until_wording():
    arrival = datetime(2026, 11, 18, 17, 50, tzinfo=LONDON)
    assert scoring.open_until_text(arrival, arrival.replace(hour=21, minute=0)) == "open until 21:00"
    assert (scoring.open_until_text(arrival, arrival.replace(hour=2, minute=0) + timedelta(days=1))
            == "open until 02:00 next day")
    assert scoring.open_until_text(arrival, arrival + timedelta(days=3)) == "open 24 hours"


# ── Backups ─────────────────────────────────────────────────────────────────


def cand(name: str, travel: int, closes: int, dlat: float, level: str = "unknown",
         price: int | None = None, rating: float | None = None) -> Candidate:
    arrival = START + timedelta(minutes=travel)
    return Candidate(
        venue=venue(f"id:{name}", name, dlat=dlat, price_level=price, rating=rating),
        travel_min=travel, arrival=arrival,
        closes_at=START.replace(hour=closes, minute=0),
        time_available_min=60 - travel, busyness=Busyness(level, "live" if level != "unknown" else "none"),
    )


def test_each_backup_has_its_own_angle_and_none_is_the_pick():
    """Acceptance 5."""
    pick = cand("Pick Place", 6, 20, 0.000, "moderate", price=2, rating=4.5)
    ranked_list = [
        pick,
        cand("Pick Place", 2, 20, 0.010),                  # same chain: a near-copy
        cand("Quick", 3, 20, 0.001),
        cand("Hush", 7, 20, 0.002, "quiet"),
        cand("Bargain", 8, 20, 0.003, price=1),
        cand("Star", 9, 20, 0.004, rating=4.9),
        cand("Late", 9, 23, 0.005),
    ]
    chosen = scoring.backups(ranked_list)
    assert [(angle, c.venue.name) for angle, c in chosen] == [
        ("closer", "Quick"), ("quieter", "Hush"), ("cheaper", "Bargain"), ("higher_rated", "Star"),
    ]
    assert len({angle for angle, _ in chosen}) == len(chosen) <= scoring.MAX_BACKUPS
    assert all(c is not pick for _, c in chosen)


def test_a_place_next_door_to_the_pick_is_not_a_backup():
    pick = cand("Pick", 6, 20, 0.0)
    twin = cand("Next Door", 2, 20, 0.0002)                # about 22 m away
    other = cand("Other", 3, 20, 0.002)
    chosen = scoring.backups([pick, twin, other])
    assert [(a, c.venue.name) for a, c in chosen][0] == ("closer", "Other")
    assert "Next Door" not in [c.venue.name for _, c in chosen]


def test_open_later_and_next_best_fill_in():
    pick = cand("Pick", 2, 19, 0.0)
    chosen = scoring.backups([pick, cand("Second", 4, 19, 0.002), cand("Owl", 5, 23, 0.004)])
    assert [(a, c.venue.name) for a, c in chosen] == [("open_later", "Owl"), ("next_best", "Second")]


def test_no_candidates_no_backups():
    assert scoring.backups([]) == []
    assert scoring.backups([cand("Only", 2, 20, 0.0)]) == []
