# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Where a memory sits must mean what the README says it means.

GitHub #136 part 2. Three corrections applied 31 seconds apart put their
successors in three different lifecycle tiers. The reporter ruled out every
retention mechanism and could not find what wrote the tier. Nothing in the
retention subsystem did: the tier came from an unseeded Langevin simulation in
the maintenance daemon, and it was noise.

README.md promises:

    "There is no retention timer counting down against a memory: what moves it
    outward is being left alone, and what pulls it back is being used."

Three measured ways the implementation did not do that:

1. ACTIVE WAS UNREACHABLE. ``r_eq = sqrt(T*dim / 2*alpha_eff)`` scales as
   sqrt(dim); the bands were dimension-independent constants. At T=0.3, dim=8
   the formula's range over the ENTIRE input domain was [0.5210, 0.6330].
   ACTIVE needed alpha_eff > 13.33 against a maximum of ~4.05, so a memory
   accessed 100,000 times at maximum importance still seeded in WARM.

2. IT WAS NOISE. Total signal across the whole metadata domain was 0.1120
   radius units. The 50-step birth burn-in alone had sd 0.1225 and a p5..p95
   spread of 0.3913 -- 3.5x the entire signal. S/N = 0.91.

3. BIRTH AGED IT. Mean radius after burn-in was 0.7515, drifting outward from
   a 0.6076 seed, p95 0.9212 -- past the ARCHIVED boundary. Fifty steps of
   simulated neglect, applied to a memory four seconds old.

The radius is now 1 - R(t), the Ebbinghaus retention this store already
computes. That makes the Langevin ball the geometric form of the forgetting
curve rather than a second, competing authority -- and it is what the README
describes. Corroboration that this was always the intended relationship: the
COLD boundary, 0.8, already equalled 1 - archive_threshold exactly.
"""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from superlocalmemory.core.maintenance import (
    _LANGEVIN_DIM,
    _seed_langevin_position,
)
from superlocalmemory.math.langevin import (
    _RADIUS_ACTIVE,
    _RADIUS_COLD,
    _RADIUS_WARM,
    LangevinDynamics,
)

_T = 0.3


def _radius(pos) -> float:
    return float(np.linalg.norm(np.asarray(pos, dtype=float)))


def _band(r: float) -> str:
    if r < _RADIUS_ACTIVE:
        return "active"
    if r < _RADIUS_WARM:
        return "warm"
    if r < _RADIUS_COLD:
        return "cold"
    return "archived"


def _seed(access: int, age_days: float, importance: float, fact_id: str = "f-1"):
    return _seed_langevin_position(
        access, age_days, importance, _T, _LANGEVIN_DIM, fact_id=fact_id,
    )


class TestANewMemoryIsBornActive:
    def test_a_brand_new_fact_lands_active(self) -> None:
        """0 of 400 reached active before this. Now it is not a sample at all."""
        r = _radius(_seed(access=0, age_days=0.0, importance=0.5))
        assert _band(r) == "active", f"a new memory was born in {_band(r)} (r={r:.4f})"

    def test_every_correction_successor_lands_in_the_same_tier(self) -> None:
        """#136's own repro: three successors, 31 seconds apart, three tiers."""
        bands = {
            _band(_radius(_seed(0, 0.0, 0.5, fact_id=f"successor-{i}")))
            for i in range(50)
        }
        assert bands == {"active"}, f"coeval successors scattered across {bands}"


class TestTheBandsAreReachable:
    @pytest.mark.parametrize(
        "access,age_days,importance,expected",
        [
            (0, 0.0, 0.5, "active"),      # written just now
            (20, 0.0, 0.9, "active"),     # written now, well used
            (0, 3650.0, 0.0, "archived"), # a decade untouched
        ],
    )
    def test_the_extremes_land_where_they_should(
        self, access, age_days, importance, expected,
    ) -> None:
        assert _band(_radius(_seed(access, age_days, importance))) == expected

    def test_more_than_one_band_is_reachable(self) -> None:
        """The old formula could only ever emit warm or cold, whatever the input."""
        seen = {
            _band(_radius(_seed(a, d, i)))
            for a in (0, 5, 100)
            for d in (0.0, 1.0, 7.0, 30.0, 365.0, 3650.0)
            for i in (0.0, 0.5, 1.0)
        }
        assert len(seen) >= 3, f"only {seen} reachable across the whole domain"


class TestLeftAloneGoesOutAndBeingUsedPullsBackIn:
    def test_being_left_alone_moves_a_memory_outward(self) -> None:
        """The README's first clause, as a monotonicity property."""
        radii = [_radius(_seed(0, d, 0.5)) for d in (0.0, 1.0, 7.0, 30.0, 365.0)]
        assert radii == sorted(radii), f"neglect did not move it outward: {radii}"

    def test_being_used_pulls_a_memory_back_in(self) -> None:
        """The README's second clause."""
        radii = [_radius(_seed(a, 30.0, 0.5)) for a in (0, 1, 5, 25, 100)]
        assert radii == sorted(radii, reverse=True), (
            f"use did not pull it back in: {radii}"
        )


class TestItIsAMeasurementRatherThanASample:
    def test_the_same_fact_always_lands_in_the_same_place(self) -> None:
        """A tier nobody can reproduce is a support case nobody can answer."""
        first = _seed(3, 10.0, 0.5, fact_id="stable-id")
        for _ in range(20):
            assert _seed(3, 10.0, 0.5, fact_id="stable-id") == first

    def test_every_new_memory_sits_at_the_centre(self) -> None:
        """r = 1 - R(0) = 0. Not a coincidence to preserve -- the point.

        Every memory is born at the origin, so there is no direction to differ
        in and nothing for thermal noise to have decided. That IS the fix for
        #136 part 2: three successors created 31 seconds apart are now the
        same point, not three samples.
        """
        assert all(
            _radius(_seed(0, 0.0, 0.5, fact_id=f"id-{i}")) == 0.0
            for i in range(25)
        )

    def test_facts_off_the_centre_get_distinct_directions(self) -> None:
        """Once they have a radius, they must not collapse onto one point.

        A shell of identical positions would make spreading activation and any
        distance-based neighbourhood meaningless.
        """
        directions = {
            tuple(round(v, 6) for v in _seed(0, 180.0, 0.5, fact_id=f"id-{i}"))
            for i in range(25)
        }
        assert len(directions) == 25

    def test_signal_now_exceeds_the_noise_it_used_to_be_buried_in(self) -> None:
        """S/N was 0.91. The measured burn-in sd it must beat was 0.1225."""
        best = _radius(_seed(100, 0.0, 1.0))
        worst = _radius(_seed(0, 3650.0, 0.0))
        signal = abs(worst - best)
        burn_in_noise_sd = 0.1225
        assert signal > 3 * burn_in_noise_sd, (
            f"signal {signal:.4f} is still within noise ({burn_in_noise_sd:.4f} sd)"
        )

    def test_a_fresh_memory_is_not_aged_by_its_own_creation(self) -> None:
        """Birth burn-in simulated 50 steps of neglect that never happened."""
        born = _radius(_seed(0, 0.0, 0.5))
        assert born < _RADIUS_ACTIVE, f"a new memory was born at r={born:.4f}"


class TestTheBandsAreDerivedNotInvented:
    def test_the_radial_bands_match_the_forgetting_thresholds(self) -> None:
        """One authority expressed in two coordinates, not two authorities."""
        from superlocalmemory.core.config import ForgettingConfig
        from superlocalmemory.math.ebbinghaus import EbbinghausCurve

        curve = EbbinghausCurve(ForgettingConfig())
        for radius, retention in (
            (_RADIUS_ACTIVE, 0.8),
            (_RADIUS_WARM, 0.5),
            (_RADIUS_COLD, ForgettingConfig().archive_threshold),
        ):
            assert radius == pytest.approx(1.0 - retention, abs=1e-9)
            # and the two classifiers agree at the boundary
            assert curve.lifecycle_zone(retention + 1e-9) in {
                "active", "warm", "cold",
            }
