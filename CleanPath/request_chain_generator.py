from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


@dataclass
class EdgeStats:
    edge_id: str
    unreachable_count: int = 0
    reachable_to: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EdgeStats":
        return cls(
            edge_id=data["edge_id"],
            unreachable_count=data.get("unreachable_count", 0),
            reachable_to=list(data.get("reachable_to", [])),
        )

    def reachable_count(self) -> int:
        return len(self.reachable_to)


class ConnectivityReport:
    def __init__(self, results: Dict[str, EdgeStats], total_candidates: int):
        self.results = results
        self.total_candidates = total_candidates
        self.edge_ids: Set[str] = set(results.keys())

    @classmethod
    def load_json(cls, path: str | Path) -> "ConnectivityReport":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = {
            edge_id: EdgeStats.from_dict(stats)
            for edge_id, stats in data["results"].items()
        }
        total_candidates = int(data.get("total_candidates", len(results)))
        return cls(results, total_candidates)

    def has_edge(self, edge_id: str) -> bool:
        return edge_id in self.results

    def stats(self, edge_id: str) -> Optional[EdgeStats]:
        return self.results.get(edge_id)

    def reachable_from(self, edge_id: str) -> List[str]:
        stats = self.results.get(edge_id)
        if stats is None:
            return []
        return stats.reachable_to

    def eligible_reachable_from(self, edge_id: str, min_reachable: int = 1) -> List[str]:
        return [
            candidate
            for candidate in self.reachable_from(edge_id)
            if self.has_edge(candidate) and self.results[candidate].reachable_count() >= min_reachable
        ]

    def top_edges_by_reachability(self, min_reachable: int = 1) -> List[str]:
        return [
            stats.edge_id
            for stats in sorted(
                (
                    s for s in self.results.values()
                    if s.reachable_count() >= min_reachable
                ),
                key=lambda s: (s.reachable_count(), -s.unreachable_count, s.edge_id),
                reverse=True,
            )
        ]


@dataclass
class TaxiAnchor:
    trip_id: str
    trip_from: Optional[str]
    trip_to: Optional[str]
    stop_edge: Optional[str]

    def ordered_edges(self, mode: str) -> List[str]:
        options: List[Optional[str]]
        if mode == "stop_first":
            options = [self.stop_edge, self.trip_to, self.trip_from]
        elif mode == "trip_to_first":
            options = [self.trip_to, self.stop_edge, self.trip_from]
        elif mode == "trip_from_first":
            options = [self.trip_from, self.trip_to, self.stop_edge]
        else:
            raise ValueError(f"Unknown anchor mode: {mode}")
        seen: Set[str] = set()
        out: List[str] = []
        for edge in options:
            if edge and edge not in seen:
                out.append(edge)
                seen.add(edge)
        return out


@dataclass
class RequestRide:
    person_id: str
    depart: float
    from_edge: str
    to_edge: str


class RequestChainGenerator:
    def __init__(self, report: ConnectivityReport, rng: random.Random):
        self.report = report
        self.rng = rng

    @staticmethod
    def _lane_to_edge(lane_id: Optional[str]) -> Optional[str]:
        if not lane_id:
            return None
        if "_" not in lane_id:
            return lane_id
        return lane_id.rsplit("_", 1)[0]

    @staticmethod
    def read_taxi_anchor(taxi_file: str | Path) -> TaxiAnchor:
        tree = ET.parse(taxi_file)
        root = tree.getroot()

        for trip in root.findall("trip"):
            trip_id = trip.get("id", "taxi_0")
            trip_from = trip.get("from")
            trip_to = trip.get("to")
            stop = trip.find("stop")
            stop_edge = None
            if stop is not None:
                stop_edge = RequestChainGenerator._lane_to_edge(stop.get("lane"))
            return TaxiAnchor(
                trip_id=trip_id,
                trip_from=trip_from,
                trip_to=trip_to,
                stop_edge=stop_edge,
            )

        raise ValueError("No <trip> element found in taxi file.")

    def _filter_existing_edges(self, edge_ids: Sequence[str]) -> List[str]:
        return [edge_id for edge_id in edge_ids if self.report.has_edge(edge_id)]

    def _choose(self, candidates: Sequence[str], label: str) -> str:
        if not candidates:
            raise ValueError(f"No candidates available for {label}.")
        return self.rng.choice(list(candidates))

    def _intersection(self, a: Sequence[str], b: Sequence[str]) -> List[str]:
        bset = set(b)
        return [x for x in a if x in bset]

    def _rank_by_reachability(self, edge_ids: Sequence[str]) -> List[str]:
        filtered = self._filter_existing_edges(edge_ids)
        return sorted(
            filtered,
            key=lambda edge_id: (
                self.report.results[edge_id].reachable_count(),
                -self.report.results[edge_id].unreachable_count,
                edge_id,
            ),
            reverse=True,
        )

    def _pick_anchor_edge(
        self,
        taxi_anchor: TaxiAnchor,
        anchor_mode: str,
        min_reachable_pickup: int,
    ) -> str:
        tried: List[str] = []
        for edge in taxi_anchor.ordered_edges(anchor_mode):
            tried.append(edge)
            if not self.report.has_edge(edge):
                continue
            if self.report.eligible_reachable_from(edge, min_reachable=min_reachable_pickup):
                return edge
        raise ValueError(
            "No usable taxi anchor edge found in the connectivity report. "
            f"Tried: {tried}. Either lower --min-reachable-pickup or regenerate the report with taxi edges included."
        )

    def _eligible_targets(self, edge_id: str, min_reachable_dropoff: int) -> List[str]:
        return self.report.eligible_reachable_from(edge_id, min_reachable=min_reachable_dropoff)

    def _sample_depart_gap(
        self,
        depart_steps: Sequence[float],
        max_random_deviation_pct: float = 0.0,
    ) -> float:
        if not depart_steps:
            raise ValueError("depart_steps must contain at least one value.")

        base_step = float(self._choose([str(step) for step in depart_steps], "depart-step"))
        if base_step < 0:
            raise ValueError("depart-step values must be non-negative.")
        if max_random_deviation_pct < 0:
            raise ValueError("max_random_deviation_pct must be non-negative.")

        if max_random_deviation_pct == 0:
            return base_step

        deviation = base_step * (max_random_deviation_pct / 100.0)
        low = max(0.0, base_step - deviation)
        high = base_step + deviation
        return self.rng.uniform(low, high)

    def generate_chain(
        self,
        num_requests: int,
        taxi_anchor: TaxiAnchor,
        anchor_mode: str = "stop_first",
        first_pool_top_k: int = 5,
        depart_start: float = 0.0,
        depart_steps: Sequence[float] = (100.0,),
        max_random_deviation_pct: float = 0.0,
        close_cycle: bool = True,
        unique_person_ids: bool = True,
        min_reachable_pickup: int = 1,
        min_reachable_dropoff: int = 1,
    ) -> tuple[str, List[RequestRide]]:
        if num_requests < 1:
            raise ValueError("num_requests must be at least 1.")
        if min_reachable_pickup < 0 or min_reachable_dropoff < 0:
            raise ValueError("Minimum reachable thresholds must be non-negative.")
        if not depart_steps:
            raise ValueError("depart_steps must contain at least one value.")
        if any(step < 0 for step in depart_steps):
            raise ValueError("All depart_steps must be non-negative.")
        if max_random_deviation_pct < 0:
            raise ValueError("max_random_deviation_pct must be non-negative.")

        anchor_edge = self._pick_anchor_edge(
            taxi_anchor=taxi_anchor,
            anchor_mode=anchor_mode,
            min_reachable_pickup=min_reachable_pickup,
        )

        rides: List[RequestRide] = []
        anchor_reachable = self.report.eligible_reachable_from(
            anchor_edge,
            min_reachable=min_reachable_pickup,
        )
        if not anchor_reachable:
            raise ValueError(
                f"Taxi anchor edge '{anchor_edge}' has no eligible pickup candidates. "
                "Try lowering --min-reachable-pickup."
            )

        top_edges = self.report.top_edges_by_reachability(min_reachable=min_reachable_pickup)
        top_edges = top_edges[: max(first_pool_top_k, 1)]
        first_from_candidates = self._intersection(anchor_reachable, top_edges)
        if not first_from_candidates:
            first_from_candidates = self._rank_by_reachability(anchor_reachable)[: max(first_pool_top_k, 1)]
        if not first_from_candidates:
            raise ValueError("No valid candidates for the first request pickup edge.")

        first_from = self._choose(first_from_candidates, "first request from-edge")
        first_to_candidates = self._eligible_targets(first_from, min_reachable_dropoff)
        if not first_to_candidates:
            raise ValueError(
                f"Chosen first pickup edge '{first_from}' has no reachable dropoff candidates that satisfy "
                f"min reachable count {min_reachable_dropoff}."
            )
        first_to = self._choose(self._rank_by_reachability(first_to_candidates), "first request to-edge")

        current_depart = depart_start
        rides.append(
            RequestRide(
                person_id="0" if unique_person_ids else "req",
                depart=current_depart,
                from_edge=first_from,
                to_edge=first_to,
            )
        )

        edges_that_can_reach_first_from = {
            edge_id
            for edge_id, stats in self.report.results.items()
            if first_from in stats.reachable_to
        }

        for idx in range(1, num_requests):
            prev_to = rides[-1].to_edge
            from_candidates = self.report.eligible_reachable_from(
                prev_to,
                min_reachable=min_reachable_pickup,
            )
            if not from_candidates:
                raise ValueError(
                    f"Previous dropoff edge '{prev_to}' has no eligible candidates for next pickup. "
                    "Try lowering --min-reachable-pickup."
                )

            ranked_from_candidates = self._rank_by_reachability(from_candidates)
            current_from = self._choose(ranked_from_candidates, f"request {idx} from-edge")
            to_candidates = self._eligible_targets(current_from, min_reachable_dropoff)
            if not to_candidates:
                raise ValueError(
                    f"Chosen pickup edge '{current_from}' has no reachable dropoff candidates satisfying "
                    f"min reachable count {min_reachable_dropoff}."
                )

            if close_cycle and idx == num_requests - 1:
                cycle_candidates = self._intersection(to_candidates, list(edges_that_can_reach_first_from))
                if cycle_candidates:
                    current_to = self._choose(
                        self._rank_by_reachability(cycle_candidates),
                        "last request to-edge (cycle closing)",
                    )
                else:
                    current_to = self._choose(
                        self._rank_by_reachability(to_candidates),
                        "last request to-edge (fallback)",
                    )
            else:
                current_to = self._choose(
                    self._rank_by_reachability(to_candidates),
                    f"request {idx} to-edge",
                )

            current_depart += self._sample_depart_gap(
                depart_steps=depart_steps,
                max_random_deviation_pct=max_random_deviation_pct,
            )

            rides.append(
                RequestRide(
                    person_id=str(idx) if unique_person_ids else "req",
                    depart=current_depart,
                    from_edge=current_from,
                    to_edge=current_to,
                )
            )

        return anchor_edge, rides

    @staticmethod
    def write_requests_file(rides: Sequence[RequestRide], output_file: str | Path) -> None:
        root = ET.Element(
            "routes",
            {
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd",
            },
        )

        for ride in rides:
            person = ET.SubElement(
                root,
                "person",
                {
                    "id": ride.person_id,
                    "depart": f"{ride.depart:.2f}",
                },
            )
            ET.SubElement(
                person,
                "ride",
                {
                    "from": ride.from_edge,
                    "to": ride.to_edge,
                    "lines": "taxi",
                },
            )

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_file, encoding="utf-8", xml_declaration=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a chained SUMO taxi request file from a connectivity report."
    )
    parser.add_argument("--report", required=True, help="Path to connectivity_report.json")
    parser.add_argument("--taxi", required=True, help="Path to taxi.rou.xml")
    parser.add_argument("--output", required=True, help="Path to output requests .rou.xml")
    parser.add_argument("--num-requests", type=int, default=36, help="Number of person requests to generate")
    parser.add_argument("--depart-start", type=float, default=0.0, help="First request depart time")
    parser.add_argument(
        "--depart-step",
        type=float,
        nargs="+",
        default=[100.0],
        help="One or more depart gaps between requests. Example: --depart-step 50 100 200",
    )
    parser.add_argument(
        "--max-random-deviation-pct",
        type=float,
        default=0.0,
        help="Maximum random deviation percentage applied to the chosen depart-step.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--first-top-k",
        type=int,
        default=5,
        help="For the first pickup, prefer candidates among the top-k most reachable eligible edges reachable from the taxi anchor",
    )
    parser.add_argument(
        "--min-reachable-pickup",
        type=int,
        default=1,
        help="Only choose pickup edges whose reachable_count is at least this value",
    )
    parser.add_argument(
        "--min-reachable-dropoff",
        type=int,
        default=1,
        help="Only choose dropoff edges whose reachable_count is at least this value",
    )
    parser.add_argument(
        "--anchor-mode",
        choices=["stop_first", "trip_to_first", "trip_from_first"],
        default="stop_first",
        help="How to choose the taxi anchor edge from taxi.rou.xml",
    )
    parser.add_argument(
        "--no-close-cycle",
        action="store_true",
        help="Disable the last-request cycle-closing preference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ConnectivityReport.load_json(args.report)
    taxi_anchor = RequestChainGenerator.read_taxi_anchor(args.taxi)

    generator = RequestChainGenerator(report, random.Random(args.seed))
    anchor_edge, rides = generator.generate_chain(
        num_requests=args.num_requests,
        taxi_anchor=taxi_anchor,
        anchor_mode=args.anchor_mode,
        first_pool_top_k=args.first_top_k,
        depart_start=args.depart_start,
        depart_steps=args.depart_step,
        max_random_deviation_pct=args.max_random_deviation_pct,
        close_cycle=not args.no_close_cycle,
        min_reachable_pickup=args.min_reachable_pickup,
        min_reachable_dropoff=args.min_reachable_dropoff,
    )
    generator.write_requests_file(rides, args.output)

    print(f"Taxi anchor edge used: {anchor_edge}")
    print(f"Generated {len(rides)} requests -> {args.output}")
    print(f"First request: {rides[0].from_edge} -> {rides[0].to_edge}")
    print(f"Last request:  {rides[-1].from_edge} -> {rides[-1].to_edge}")
    print(
        "Thresholds used: "
        f"pickup >= {args.min_reachable_pickup}, "
        f"dropoff >= {args.min_reachable_dropoff}"
    )


if __name__ == "__main__":
    main()
