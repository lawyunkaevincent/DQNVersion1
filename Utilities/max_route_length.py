import xml.etree.ElementTree as ET
import sumolib

net_file = "D:\\6Sumo\\RLTesting\\RLTrainingMap1\\map.net.xml"
net = sumolib.net.readNet(net_file)
mx_len, mx_tag = 0.0, None
root = ET.parse("routes.rou.xml").getroot()

def route_len(edge_ids):
    return sum(net.getEdge(e).getLength() for e in edge_ids if not e.startswith(":"))

# <route> elements
for r in root.iter("route"):
    edges = r.get("edges", "").split()
    L = route_len(edges)
    if L > mx_len:
        mx_len, mx_tag = L, f"route id={r.get('id')}"

# <vehicle route='...'> shorthand
for v in root.iter("vehicle"):
    r = v.find("route")
    if r is not None and r.get("edges"):
        edges = r.get("edges").split()
        L = route_len(edges)
        if L > mx_len:
            mx_len, mx_tag = L, f"vehicle id={v.get('id')}"

print("max file route length:", mx_len, "for", mx_tag)
