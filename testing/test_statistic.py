# run_sumo_log_state.py
import os
import sys
import argparse
import time

def ensure_sumo_on_path():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        sys.path.append(tools)
    else:
        print("ERROR: Please set SUMO_HOME (e.g., export SUMO_HOME=/path/to/sumo).")
        sys.exit(1)

def main():
    ensure_sumo_on_path()
    import traci  # after SUMO tools path is added

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

    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # --- Minimal “state” snapshot (safe calls only) ---
            sim_time = traci.simulation.getTime()
            veh_ids = traci.vehicle.getIDList()
            per_ids = traci.person.getIDList()
            if per_ids:
                wait_times = [traci.person.getWaitingTime(pid) for pid in per_ids]
                pid_max_wait = per_ids[wait_times.index(max(wait_times))]
                oldest_wait = max(wait_times)
            else:
                oldest_wait = 0
            num_veh = len(veh_ids)
            num_persons = len(per_ids)
            edge_ids = traci.edge.getIDList()
            if edge_ids:
                mean_speeds = [traci.edge.getLastStepMeanSpeed(e) for e in edge_ids if not e.startswith(":")]
                net_mean_speed = sum(mean_speeds) / max(1, len(mean_speeds))
            else:
                net_mean_speed = -1.0
            # requests = traci.person.getTaxiReservations(0)
            # print(requests)
            state = {
                "time": sim_time,
                "num_veh": num_veh,
                "num_persons": num_persons,
                "longest wait time": oldest_wait,
                "longest_wait_pid": pid_max_wait if per_ids else "N/A",
                "net_mean_speed": round(net_mean_speed, 3),
            }

            line = f"{state}"
            print(line)
            if log_f:
                log_f.write(line + "\n")

        # One final step to flush events (optional)
        # traci.simulationStep()

    finally:
        if log_f:
            log_f.close()
        traci.close()

if __name__ == "__main__":
    main()
