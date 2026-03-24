import xml.etree.ElementTree as ET
import random

# === Settings ===
DEPART_RULE = "random"  # options: "random", "scale", "sequential"
RANDOM_RANGE = (0, 100)  # seconds if using random
SCALE_FACTOR = 80.0      # if using scale
SEQUENTIAL_OFFSET = 10   # if using sequential (each +5s)

master_xml = r"D:\\6Sumo\\RLTesting\\RLTrainingMap2\\persontrips.xml"
generated_xml = r"D:\\6Sumo\\RLTesting\\RLTrainingMap2\\persontripsridecheck.rou.xml"

# === Load files ===
tree_ref = ET.parse(master_xml)       # master (personTrip)
root_ref = tree_ref.getroot()

tree_walk = ET.parse(generated_xml)  # generated (walk/ride)
root_walk = tree_walk.getroot()

# === Function to update depart times ===
def update_depart(old, pid):
    if DEPART_RULE == "random":
        return old*SCALE_FACTOR + random.uniform(*RANDOM_RANGE)
    elif DEPART_RULE == "scale":
        return old * SCALE_FACTOR
    elif DEPART_RULE == "sequential":
        return pid * SEQUENTIAL_OFFSET
    else:
        return old  # default, no change

# === Process persons ===
for p_ref, p_walk in zip(root_ref.findall("person"), root_walk.findall("person")):
    pid = int(p_ref.get("id"))
    # print(pid)
    old_depart = float(p_ref.get("depart"))
    print(f"old_depart: {old_depart}, pid: {pid}")
    new_depart = update_depart(old_depart, pid)

    # --- Update depart in BOTH files ---
    p_ref.set("depart", f"{new_depart:.2f}")
    p_walk.set("depart", f"{new_depart:.2f}")

    # --- Replace <walk> in .rou with <ride> ---
    for child in list(p_walk):
        p_walk.remove(child)

    trip = p_ref.find("personTrip")
    if trip is not None:
        from_edge = trip.get("from")
        to_edge = trip.get("to")
        ride = ET.Element("ride", {
            "from": from_edge,
            "to": to_edge,
            "lines": "taxi"
        })
        p_walk.append(ride)

# === Save outputs ===
master_xml = r"D:\\6Sumo\\RLTesting\\RLTrainingMap2\\persontrips_scale.xml"
generated_xml = r"D:\\6Sumo\\RLTesting\\RLTrainingMap2\\persontrips_scale.rou.xml"
tree_ref.write(master_xml, encoding="UTF-8", xml_declaration=True)
tree_walk.write(generated_xml, encoding="UTF-8", xml_declaration=True)

print("✅ Updated both persontrips.xml and persontrips.rou.xml")
