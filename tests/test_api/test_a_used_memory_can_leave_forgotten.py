# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""`archive` and `forgotten` must not be one-way doors.

The demotion ladder in `/api/v3/forgetting/run` excludes them so a deliberately
retired memory is not churned. The PROMOTION step excluded them too — which
meant a memory being actively used could never climb back.

That turned a calculation error into a permanent one. When the decay curve
(parameterised in hours, fed real elapsed time) filed **5,546 of 5,561
memories** as `forgotten`, correcting the curve fixed nothing: every one of
them was already behind the door.

README: "what moves it outward is being left alone, and what pulls it back is
being used." The second half needs this.
"""

from __future__ import annotations

import inspect

from superlocalmemory.server.routes import v3_api


def test_promotion_does_not_exclude_archive_or_forgotten() -> None:
    src = inspect.getsource(v3_api)
    promote = src[src.index("SET lifecycle_zone = 'active' "):]
    stmt = promote[:promote.index(")")]
    assert "NOT IN ('archive', 'forgotten')" not in stmt, (
        "a used memory still cannot climb out of archive/forgotten"
    )


def test_the_demotion_ladder_still_excludes_them() -> None:
    """Demotion must stay one-way, or retired memories churn every cycle."""
    src = inspect.getsource(v3_api)
    ladder = src[src.index("for zone, threshold in zone_thresholds:"):]
    assert "NOT IN ('archive', 'forgotten')" in ladder[:900]
