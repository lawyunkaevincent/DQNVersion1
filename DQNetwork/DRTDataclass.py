# drt_dataclasses.py
"""
Core dataclasses for the shared-ride DRT environment.
These replace the ad-hoc dicts and positional variables scattered across SUMOENV.py.

Design decisions reflected here:
  - Periodic 10-second ticks; only MEANINGFUL ticks (non-empty candidate set)
    become RL transitions stored in the replay buffer.
  - No request rejection — every passenger must eventually be served.
    DEFER is allowed (postpone to next tick) but not reject.
  - Edge-only spatial representation matching the route XML (no position offsets).
  - Each RL action = one CandidateInsertion chosen from a dynamic feasible set.
  - NN scores each candidate independently with a fixed-length feature vector.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RequestStatus(Enum):
    PENDING   = auto()   # arrived, not yet assigned to any taxi
    ASSIGNED  = auto()   # assigned to a taxi, waiting for pickup
    ONBOARD   = auto()   # picked up, en route to dropoff
    COMPLETED = auto()   # dropped off successfully
    DEFERRED  = auto()   # agent chose DEFER; still in pending pool
    # NOTE: rejection is not used — every request must eventually be served


class StopType(Enum):
    PICKUP  = auto()
    DROPOFF = auto()


class TaxiStatus(Enum):
    IDLE     = auto()   # no passengers, no assignment  (SUMO fleet state 0)
    PICKUP   = auto()   # driving to pick up someone    (SUMO fleet state 1)
    OCCUPIED = auto()   # has onboard passenger(s)      (SUMO fleet state 2)
    PICKUP_AND_OCCUPIED = auto()  # both                (SUMO fleet state 3)


class TickOutcome(Enum):
    """
    Result of evaluating a 10-second periodic tick.

    Every tick the environment checks whether anything meaningful has changed.
    Only MEANINGFUL ticks are stored as RL transitions in the replay buffer;
    IDLE ticks are simulated through without creating a training sample.

    MEANINGFUL  : at least one pending/deferred request exists AND at least
                  one taxi has capacity → candidate set is non-empty, agent acts.
    IDLE        : no actionable state (e.g. all requests assigned, all taxis
                  full, or no requests in system yet) → skip RL step, keep
                  accumulating reward in the current interval.
    """
    MEANINGFUL = auto()
    IDLE       = auto()


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """
    One passenger trip request.

    IDs
    ---
    request_id      : TraCI reservation ID  (str, e.g. "0")
    person_id       : TraCI person ID       (str, e.g. "0")
                      In your XML, person id and the reservation id that SUMO
                      assigns share the same base identifier.
    Both are stored because SUMO uses them in different API calls:
      - dispatchTaxi()            uses the reservation ID
      - person.getWaitingTime()   uses the person ID

    Spatial
    -------
    from_edge / to_edge : SUMO edge IDs taken directly from the <ride> tag.
                          No position offsets are used — your route file only
                          specifies edges (e.g. from="E14" to="E17"), so SUMO
                          places the person at the default position on that edge.

    Timing
    ------
    request_time      : sim time when the person departed (depart= in XML)
    direct_travel_time: findRoute(from_edge → to_edge).travelTime
                        computed once when request first appears in TraCI;
                        used as the solo-service reference for detour penalty
    pickup_time       : sim time when passenger was actually picked up  (None until then)
    dropoff_time      : sim time when passenger was dropped off         (None until then)

    Assignment
    ----------
    assigned_taxi_id  : which taxi this request is committed to  (None if PENDING/DEFERRED)
    status            : RequestStatus enum

    Constraints
    -----------
    max_wait          : hard upper bound on pre-pickup waiting (seconds)
    max_ride_factor   : actual_ride_time must be ≤ max_ride_factor × direct_travel_time
    """

    # --- identity ---
    request_id: str
    person_id:  str

    # --- spatial (edge IDs only, no position offsets) ---
    from_edge: str
    to_edge:   str

    # --- timing ---
    request_time:       float = 0.0
    direct_travel_time: float = 0.0   # set once on first appearance via findRoute
    pickup_time:        Optional[float] = None
    dropoff_time:       Optional[float] = None

    # --- assignment ---
    assigned_taxi_id: Optional[str] = None
    status: RequestStatus = RequestStatus.PENDING

    # --- service constraints ---
    max_wait:        float = 600.0   # 10 min default; tune per scenario
    max_ride_factor: float = 2     # actual ride ≤ 2× direct travel time

    # --- derived helpers ---
    def waiting_time(self, now: float) -> float:
        """Seconds this request has been waiting for pickup."""
        return now - self.request_time

    def slack_to_max_wait(self, now: float) -> float:
        """Remaining seconds before max_wait is breached. Negative = already violated."""
        return self.max_wait - self.waiting_time(now)

    @property
    def excess_ride_time(self) -> Optional[float]:
        """Detour burden on this passenger. None until dropped off."""
        if self.pickup_time is None or self.dropoff_time is None:
            return None
        actual = self.dropoff_time - self.pickup_time
        return max(0.0, actual - self.direct_travel_time)

    @property
    def is_active(self) -> bool:
        """True if still in the system and needing service."""
        return self.status in (RequestStatus.PENDING,
                               RequestStatus.ASSIGNED,
                               RequestStatus.ONBOARD,
                               RequestStatus.DEFERRED)


# ---------------------------------------------------------------------------
# Stop  (one entry in a taxi's future route plan)
# ---------------------------------------------------------------------------

@dataclass
class Stop:
    """
    A single future stop in a taxi's planned route.

    stop_type  : PICKUP or DROPOFF
    request_id : which request this stop belongs to
    person_id  : SUMO person ID (needed for dispatchTaxi calls)
    edge_id    : SUMO edge where the stop is located.
                 Taken directly from the request's from_edge / to_edge —
                 no position offset is stored, matching the route file format.
    eta        : estimated sim-time arrival at this stop.
                 Recomputed after every insertion via findRoute chains.
    """
    stop_type:  StopType
    request_id: str
    person_id:  str
    edge_id:    str
    eta:        float = 0.0   # recomputed after each insertion

    def __repr__(self) -> str:
        tag = "PU" if self.stop_type == StopType.PICKUP else "DO"
        return f"Stop({tag} req={self.request_id} eta={self.eta:.1f}s)"


# ---------------------------------------------------------------------------
# TaxiPlan  (full state of one taxi)
# ---------------------------------------------------------------------------

@dataclass
class TaxiPlan:
    """
    Everything we track about one taxi.

    Identity
    --------
    taxi_id      : SUMO vehicle ID

    Position
    --------
    current_edge : SUMO road ID where the taxi currently is
    current_x/y  : Cartesian coordinates (from traci.vehicle.getPosition)
                   Used for distance/ETA estimates in candidate generation.

    Capacity
    --------
    capacity     : max number of onboard passengers
    onboard_count: passengers currently inside

    Status
    ------
    status       : TaxiStatus enum (mirrors SUMO fleet state)

    Route plan
    ----------
    stops        : ordered list of future Stop objects.
                   The taxi will visit them in this sequence.
                   Invariant: every PICKUP for request r appears before
                   its corresponding DROPOFF.

    Bookkeeping
    -----------
    assigned_request_ids : set of request IDs committed to this taxi
                           (both already-onboard and still-to-pick-up)
    onboard_request_ids  : subset currently inside the vehicle

    Timing / cost estimates
    -----------------------
    remaining_route_time : total estimated time to complete all stops
    next_stop_eta        : ETA to the immediately next stop
    cumulative_distance  : traci.vehicle.getDistance() snapshot
                           used to compute empty-driving increments
    """

    # --- identity ---
    taxi_id: str

    # --- position ---
    current_edge: str   = ""
    # current_x:    float = 0.0
    # current_y:    float = 0.0

    # --- capacity ---
    capacity:      int = 10
    onboard_count: int = 0

    # --- status ---
    status: TaxiStatus = TaxiStatus.IDLE

    # --- route plan ---
    stops: list[Stop] = field(default_factory=list)

    # --- bookkeeping ---
    assigned_request_ids: set[str] = field(default_factory=set)
    onboard_request_ids:  set[str] = field(default_factory=set)

    # --- timing / cost ---
    remaining_route_time: float = 0.0
    next_stop_eta:        float = 0.0
    cumulative_distance:  float = 0.0   # updated from SUMO each step

    # --- derived helpers ---
    @property
    def remaining_capacity(self) -> int:
        return self.capacity - self.onboard_count

    @property
    def num_future_stops(self) -> int:
        return len(self.stops)

    @property
    def is_idle(self) -> bool:
        return self.status == TaxiStatus.IDLE

    @property
    def has_capacity(self) -> bool:
        return self.remaining_capacity > 0

    def pickup_index_for(self, request_id: str) -> Optional[int]:
        """Return the index of the PICKUP stop for request_id, or None."""
        for i, s in enumerate(self.stops):
            if s.request_id == request_id and s.stop_type == StopType.PICKUP:
                return i
        return None

    def dropoff_index_for(self, request_id: str) -> Optional[int]:
        """Return the index of the DROPOFF stop for request_id, or None."""
        for i, s in enumerate(self.stops):
            if s.request_id == request_id and s.stop_type == StopType.DROPOFF:
                return i
        return None

    def __repr__(self) -> str:
        stops_str = " → ".join(repr(s) for s in self.stops)
        return (f"TaxiPlan(id={self.taxi_id} "
                f"load={self.onboard_count}/{self.capacity} "
                f"status={self.status.name} "
                f"stops=[{stops_str}])")


# ---------------------------------------------------------------------------
# CandidateInsertion  (one possible RL action)
# ---------------------------------------------------------------------------

@dataclass
class CandidateInsertion:
    """
    One feasible way to insert a request into a taxi's route.

    This is the unit the NN scores; the agent picks the best one.

    Identity
    --------
    request_id    : request being inserted
    taxi_id       : taxi receiving the insertion
    pickup_index  : position in taxi.stops where the PICKUP stop is inserted
    dropoff_index : position after insertion where the DROPOFF stop lands
                    (dropoff_index > pickup_index always)

    Resulting plan
    --------------
    resulting_stops : the full new stop list after insertion
                      (used to apply the action to SUMO)

    Cost / feasibility estimates
    ----------------------------
    added_route_time          : extra seconds added to taxi's total route
    added_travel_distance     : extra metres
    pickup_eta_new            : predicted sim time the new passenger is picked up
    dropoff_eta_new           : predicted sim time the new passenger is dropped off
    max_existing_delay        : worst-case extra delay imposed on already-assigned passengers
    avg_existing_delay        : average extra delay across existing passengers
    min_remaining_slack       : smallest remaining time-slack before any constraint is violated
    is_feasible               : True if all hard constraints are satisfied
    infeasibility_reason      : human-readable reason if not feasible

    Special flags
    -------------
    is_no_op   : True for the NO_OP / KEEP_CURRENT_PLANS pseudo-action
    is_defer   : True for the DEFER(request) pseudo-action
    """

    # --- identity ---
    request_id:   str
    taxi_id:      str
    pickup_index: int
    dropoff_index: int

    # --- resulting plan ---
    resulting_stops: list[Stop] = field(default_factory=list)

    # --- cost estimates ---
    added_route_time:      float = 0.0
    added_travel_distance: float = 0.0
    pickup_eta_new:        float = 0.0
    dropoff_eta_new:       float = 0.0
    max_existing_delay:    float = 0.0
    avg_existing_delay:    float = 0.0
    max_pickup_delay:      float = 0.0  # pickup delay of existing request due to the new insertion
    new_wait_violation: float = 0.0
    new_ride_violation: float = 0.0
    existing_wait_violation_sum: float = 0.0
    existing_wait_violation_max: float = 0.0
    existing_ride_violation_sum: float = 0.0
    existing_ride_violation_max: float = 0.0
    min_remaining_slack:   float = float("inf")

    # --- feasibility ---
    is_feasible:          bool = True
    infeasibility_reason: str  = ""

    # --- pseudo-action flags ---
    # NOTE: is_no_op is removed — under the periodic design, if there are
    # no pending requests the tick is classified IDLE and no RL step occurs.
    # DEFER is kept: agent may choose to delay a request to a future tick.
    is_defer: bool = False

    @classmethod
    def make_defer(cls, request_id: str) -> "CandidateInsertion":
        """
        Factory for the DEFER pseudo-action.
        The request stays in the pending pool and will be reconsidered
        at the next MEANINGFUL tick. Every request must eventually be served —
        DEFER is not rejection, just postponement.
        """
        c = cls(request_id=request_id, taxi_id="", pickup_index=-1, dropoff_index=-1)
        c.is_defer = True
        c.is_feasible = True
        return c

    def __repr__(self) -> str:
        if self.is_defer:
            return f"Candidate(DEFER req={self.request_id})"
        return (f"Candidate(req={self.request_id} taxi={self.taxi_id} "
                f"pu_idx={self.pickup_index} do_idx={self.dropoff_index} "
                f"+time={self.added_route_time:.1f}s "
                f"feasible={self.is_feasible})")


# ---------------------------------------------------------------------------
# GlobalStateSummary  (system-wide features for the NN)
# ---------------------------------------------------------------------------

@dataclass
class GlobalStateSummary:
    """
    Snapshot of overall system state at a decision event.
    All fields are used as NN input features (after normalisation).
    """
    sim_time:            float = 0.0
    pending_req_count:   int   = 0    # PENDING + DEFERRED
    onboard_count:       int   = 0    # passengers currently inside a taxi
    idle_taxi_count:     int   = 0
    active_taxi_count:   int   = 0
    avg_wait_time:       float = 0.0  # mean over all PENDING/DEFERRED requests
    max_wait_time:       float = 0.0  # oldest unserved request
    avg_occupancy:       float = 0.0  # mean onboard_count / capacity across fleet
    fleet_utilization:   float = 0.0  # fraction of taxis not idle
    recent_demand_rate:  float = 0.0  # requests arrived in last 60 s


# ---------------------------------------------------------------------------
# TickContext  (metadata about the current 10-second tick)
# ---------------------------------------------------------------------------

@dataclass
class TickContext:
    """
    Metadata attached to each periodic 10-second tick evaluation.

    outcome          : MEANINGFUL (agent acts, transition stored) or
                       IDLE (nothing actionable, tick is skipped for RL)
    pending_pool     : snapshot of request IDs that are PENDING or DEFERRED
                       at this tick — these are the requests the agent can
                       choose to insert or defer again
    has_candidates   : True if at least one feasible CandidateInsertion exists
                       (pre-computed before deciding whether to call the NN)
    sim_time         : simulation time at this tick
    """
    outcome:       TickOutcome = TickOutcome.IDLE
    pending_pool:  list[str]   = field(default_factory=list)   # request_ids
    has_candidates: bool       = False
    sim_time:      float       = 0.0


# ---------------------------------------------------------------------------
# IntervalAccumulator  (reward bookkeeping between decisions)
# ---------------------------------------------------------------------------

@dataclass
class IntervalAccumulator:
    """
    Accumulates service costs between two consecutive RL decisions.
    Reset each time the agent acts; converted to reward at the next event.

    wait_cost        : sum of (num_waiting_passengers × dt) each sim step
                       i.e. total passenger-seconds of pre-pickup waiting
    ride_cost        : sum of excess in-vehicle time for passengers dropped
                       off during this interval
    empty_dist_cost  : total metres driven by taxis with no onboard passenger
    completed_dropoffs: count of passengers dropped off in this interval
    violations       : count of hard-constraint breaches (wait > max_wait, etc.)
    elapsed_time     : wall-clock sim seconds from action to next decision
    """
    wait_cost:          float = 0.0
    ride_cost:          float = 0.0
    empty_dist_cost:    float = 0.0
    completed_dropoffs: int   = 0
    violations:         int   = 0
    elapsed_time:       float = 0.0

    def reset(self) -> None:
        self.wait_cost          = 0.0
        self.ride_cost          = 0.0
        self.empty_dist_cost    = 0.0
        self.completed_dropoffs = 0
        self.violations         = 0
        self.elapsed_time       = 0.0

    def compute_reward(self,
                       w_wait:     float = 0.01,
                       w_ride:     float = 0.02,
                       w_empty:    float = 0.005,
                       w_complete: float = 2.0,
                       w_violate:  float = 1.0) -> float:
        """
        Convert accumulated costs into a scalar reward.

        Signs:
          - waiting, detour, empty driving → negative (penalties)
          - completions → positive (bonus)
          - violations  → strongly negative

        Default weights are starting points; tune per scenario.
        """
        return (
            - w_wait     * self.wait_cost
            - w_ride     * self.ride_cost
            - w_empty    * self.empty_dist_cost
            + w_complete * self.completed_dropoffs
            - w_violate  * self.violations
        )

    def __repr__(self) -> str:
        return (f"Accumulator("
                f"wait={self.wait_cost:.1f}ps "
                f"ride={self.ride_cost:.1f}s "
                f"empty={self.empty_dist_cost:.1f}m "
                f"done={self.completed_dropoffs} "
                f"violations={self.violations} "
                f"dt={self.elapsed_time:.1f}s)")