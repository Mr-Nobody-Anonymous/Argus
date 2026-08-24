"""
CityOS corridor intelligence - cross-intersection track continuity.

Deployment registration (`bind_camera`) says which cameras cover which
intersection. This layer answers a different question: did the vehicle that
left intersection A heading east subsequently arrive at intersection B from
the west, within a physically plausible travel time?

Matches are CANDIDATES, never identifications: two similar vehicles could
both make the trip. Every match carries its uncertainty (time window, route
plausibility) and `is_identification` stays false - the same honesty rule the
appearance-search layer uses.

    Intersection A --(exit approach)--> corridor --> (entry approach) B
"""

import threading
import time
from collections import deque
from typing import Dict, List, Optional


class CorridorLink:
    """A directed connection between two intersections."""

    __slots__ = ("link_id", "from_iid", "to_iid", "exit_approach",
                 "entry_approach", "min_travel_s", "max_travel_s")

    def __init__(self, link_id: str, from_iid: str, to_iid: str,
                 exit_approach: str, entry_approach: str,
                 min_travel_s: float = 20.0, max_travel_s: float = 300.0):
        self.link_id = link_id
        self.from_iid = from_iid
        self.to_iid = to_iid
        self.exit_approach = exit_approach      # approach at A facing B
        self.entry_approach = entry_approach    # approach at B facing A
        self.min_travel_s = float(min_travel_s)
        self.max_travel_s = float(max_travel_s)

    def to_dict(self) -> Dict:
        return {
            "link_id": self.link_id,
            "from_intersection": self.from_iid,
            "to_intersection": self.to_iid,
            "exit_approach": self.exit_approach,
            "entry_approach": self.entry_approach,
            "travel_time_s": [self.min_travel_s, self.max_travel_s],
        }


class CorridorService:
    """Track-continuity candidates across linked intersections."""

    MAX_PENDING = 2000
    PENDING_TTL_S = 600.0

    def __init__(self):
        self._lock = threading.Lock()
        self.links: Dict[str, CorridorLink] = {}
        # pending handoffs: list of {track_id, link_id, ts}
        self._pending: List[Dict] = []
        self.matches: deque = deque(maxlen=500)
        self.travel_times: Dict[str, deque] = {}   # link_id -> samples

    # ── Configuration ───────────────────────────────────────────────────

    def add_link(self, link: CorridorLink):
        with self._lock:
            self.links[link.link_id] = link

    def remove_link(self, link_id: str):
        with self._lock:
            self.links.pop(link_id, None)

    # ── Ingest ──────────────────────────────────────────────────────────

    def register_exit(self, iid: str, trip: Dict):
        """A completed trip at an intersection may continue down a corridor."""
        now = time.time()
        with self._lock:
            for link in self.links.values():
                if link.from_iid != iid:
                    continue
                if trip.get("exit") != link.exit_approach:
                    continue
                self._pending.append({
                    "track_id": str(trip.get("track_id")),
                    "category": trip.get("category"),
                    "link_id": link.link_id,
                    "ts": now,
                })
            # Bound + expire.
            if len(self._pending) > self.MAX_PENDING:
                del self._pending[:-self.MAX_PENDING]
            cutoff = now - self.PENDING_TTL_S
            self._pending = [p for p in self._pending if p["ts"] > cutoff]

    def register_entry(self, iid: str, user: Dict) -> Optional[Dict]:
        """Try to match a newly-seen object against pending handoffs."""
        now = time.time()
        with self._lock:
            best = None
            best_dt = None
            for p in self._pending:
                link = self.links.get(p["link_id"])
                if link is None or link.to_iid != iid:
                    continue
                if user.get("approach") != link.entry_approach:
                    continue
                if user.get("category") != p["category"]:
                    continue
                dt = now - p["ts"]
                if not (link.min_travel_s <= dt <= link.max_travel_s):
                    continue
                if best_dt is None or abs(dt - (link.min_travel_s +
                                                link.max_travel_s) / 2) < \
                        abs(best_dt - (link.min_travel_s +
                                       link.max_travel_s) / 2):
                    best, best_dt = p, dt
            if best is None:
                return None

            # Consume the handoff: one departure explains one arrival.
            self._pending.remove(best)
            link = self.links[best["link_id"]]
            sample = {
                "match_id": f"{best['link_id']}-{int(now*1000)}",
                "link_id": best["link_id"],
                "from_track_id": best["track_id"],
                "to_track_id": str(user.get("track_id")),
                "travel_time_s": round(best_dt, 1),
                "is_identification": False,
                "caveat": ("corridor match is a candidate based on timing and "
                           "route geometry, not a confirmed identity"),
                "timestamp": round(now, 3),
            }
            self.matches.append(sample)
            self.travel_times.setdefault(best["link_id"], deque(maxlen=200))\
                .append(best_dt)
            return sample

    # ── Queries ─────────────────────────────────────────────────────────

    def recent_matches(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            out = list(self.matches)[-limit:]
        out.reverse()
        return out

    def travel_time_stats(self) -> Dict[str, Dict]:
        with self._lock:
            out = {}
            for link_id, samples in self.travel_times.items():
                vals = sorted(samples)
                if not vals:
                    continue
                out[link_id] = {
                    "samples": len(vals),
                    "median_s": round(vals[len(vals) // 2], 1),
                    "min_s": round(vals[0], 1),
                    "max_s": round(vals[-1], 1),
                }
            return out

    def summary(self) -> Dict:
        with self._lock:
            return {
                "links": [l.to_dict() for l in self.links.values()],
                "pending_handoffs": len(self._pending),
                "matches_recorded": len(self.matches),
                "travel_time_stats": self.travel_time_stats(),
            }