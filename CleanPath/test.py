import xml.etree.ElementTree as ET

rou_path = "persontrips_scale.rou.xml"   # change to your file path

tree = ET.parse(rou_path)
root = tree.getroot()
print(root)

# records = []
# for person in root.findall("person"):
#     pid = person.get("id")
#     depart = float(person.get("depart", "0"))

#     ride = person.find("ride")
#     if ride is None:
#         # if you later have <walk> persons, handle them here
#         continue

#     records.append({
#         "person_id": pid,
#         "depart": depart,
#         "from_edge": ride.get("from"),
#         "to_edge": ride.get("to"),
#         "lines": ride.get("lines"),
#     })

# print(records[:3])