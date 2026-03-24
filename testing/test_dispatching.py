# run_sumo_log_state.py
import os
import sys
import argparse
import time
import traci  

def ensure_sumo_on_path():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        print("ERROR: Please set SUMO_HOME (e.g., export SUMO_HOME=/path/to/sumo).")
        sys.exit(1)

def get_oldest_request(requests):
        max_wait = -1
        for req in requests:
            wait_time = traci.person.getWaitingTime(req.persons[0])
            max_wait = max(max_wait, wait_time)
        return max_wait if max_wait >= 0 else 0

def main():
    ensure_sumo_on_path()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Path to .sumocfg")
    parser.add_argument("--gui", action="store_true", help="Use sumo-gui")
    parser.add_argument("--step-length", type=float, default=1.0, help="SUMO step length (s)")
    parser.add_argument("--log", default="", help="Optional path to save states (txt)")
    args = parser.parse_args()

    sumo_bin = "sumo-gui" if args.gui else "sumo"
    sumo_cmd = [sumo_bin, "-c", args.cfg, "--step-length", str(args.step_length)]

    traci.start(sumo_cmd)
    log_f = open(args.log, "w") if args.log else None
    count = 0
    try:
        assign = False
        taxi_fetch = False
        delay_count = 50
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            count += 1

            # --- Minimal “state” snapshot (safe calls only) ---
            sim_time = traci.simulation.getTime()
            veh_ids = traci.vehicle.getIDList()
            per_ids = traci.person.getIDList()
            # pending ride requests (person IDs waiting)
            requests = traci.person.getTaxiReservations(0)  # 0 = pending only
            taxis = traci.vehicle.getTaxiFleet(0)  # all idle taxis
            print(f"----------------TIMESTEP {sim_time}--------------------------")
            print(f"Requests [{requests}]")
            print(f"Idle taxi [{taxis}]")
            if len(requests) == 2 and not assign:
                delay_count -= 1
                if delay_count == 0:
                    vType = traci.vehicle.getTypeID(taxis[0])
                    taxi0loc = traci.vehicle.getRoadID(taxis[0])
                    taxi1loc = traci.vehicle.getRoadID(taxis[1])
                    request0loc = requests[0].fromEdge
                    request1loc = requests[1].fromEdge
                    # start information testing
                    dest = requests[0].toEdge



                    # end information testing
                    r00 = traci.simulation.findRoute(taxi0loc, request0loc, vType, routingMode=1).travelTime
                    r01 = traci.simulation.findRoute(taxi0loc, request1loc, vType, routingMode=1).travelTime
                    r10 = traci.simulation.findRoute(taxi1loc, request0loc, vType, routingMode=1).travelTime
                    r11 = traci.simulation.findRoute(taxi1loc, request1loc, vType, routingMode=1).travelTime
                    # costs to pickup
                    c00, c01 = r00, r01
                    c10, c11 = r10, r11
                    # choose pairing that minimizes the slower pickup
                    diag_max = max(c00, c11)
                    cross_max = max(c01, c10)
                    if diag_max <= cross_max:
                        print("Assign taxi 0 to request 0 & taxi 1 to request 1")
                        traci.vehicle.dispatchTaxi(taxis[0], requests[0].id)
                        traci.vehicle.dispatchTaxi(taxis[1], requests[1].id)
                        assign = True
                    else:    
                        print("Assign taxi 1 to request 0 & taxi 0 to request 1")
                        traci.vehicle.dispatchTaxi(taxis[1], requests[0].id)
                        traci.vehicle.dispatchTaxi(taxis[0], requests[1].id)
                        assign = True

                    print("\n\n")
                    print(f"Person Type in Requests: {type(requests[0].persons)}")
                    print(f"Person in Requests: {requests[0].persons}")
                    print(f"Longest wait time: {get_oldest_request(requests)}")
                    print(traci.vehicle.getNextStops(taxis[1]))
                    print(traci.vehicle.getStops(taxis[1]))
                    print("\n\n")
                    break

            if len(traci.person.getTaxiReservations(8)) == 2:
                taxi_fetch = True
            if taxi_fetch:
                if len(traci.vehicle.getTaxiFleet(0)) == 2:
                    break


    finally:
        if log_f:
            log_f.close()
        traci.close()

if __name__ == "__main__":
    main()
