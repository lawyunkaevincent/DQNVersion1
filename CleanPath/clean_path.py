from __future__ import annotations
import xml.etree.ElementTree as ET
import sumolib.net  
import shutil

class Edge: 
    def __init__(self, id):
        self.id = id
        self.unreachable_edge = []
        self.unreachable_count = 0
        
    def get_id(self):
        return self.id
    
    def add_unreachable_edge(self, edge: Edge):
        self.unreachable_edge.append(edge)
        
    def get_unreachable_edges(self):
        return self.unreachable_edge
    
    def add_unreachable_count(self):
        self.unreachable_count += 1
        
    def get_unreachable_count(self):
        return self.unreachable_count
        
    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Edge) and self.id == other.id
    
    def __str__(self):
        return f"{self.id} ({self.unreachable_count})"
    
    
class ReqReader:
    def __init__(self, route_file, net_file):
        self.tree = ET.parse(route_file)
        self.route_file = route_file
        self.net = sumolib.net.readNet(net_file)
        self.edge_set = set()
        
    def read_edge(self):
        root = self.tree.getroot()
        for person in root.findall("person"):
            ride = person.find("ride")
            if ride is None:
                # if you later have <walk> persons, handle them here
                continue
            from_edge = Edge(ride.get("from")) 
            to_edge = Edge(ride.get("to"))
            self.edge_set.add(from_edge)
            self.edge_set.add(to_edge)
            
    def check_reachability(self):
        for edge in self.edge_set:
            # print(f"Checking reachability from edge {edge}")
            for other_edge in self.edge_set:
                if edge != other_edge:
                    # print(f"  Checking reachability to edge {other_edge}")
                    start = self.net.getEdge(edge.get_id())
                    end = self.net.getEdge(other_edge.get_id())
                    path, cost = self.net.getShortestPath(start, end)
                    if path:
                        ok = all(e.allows("passenger") for e in path)  # often taxis behave like passenger
                        if not ok:
                            edge.add_unreachable_edge(other_edge)
                            other_edge.add_unreachable_count()
                            # print(f"    Edge {other_edge} is unreachable from {edge}")
                            
            # print(f"Total unreachable edges from {edge}: {len(edge.get_unreachable_edges())}/{len(self.edge_set)}\n")
                            
    def unreachable_edges_report(self):
        sorted_edges = sorted(self.edge_set, key=lambda e: e.get_unreachable_count(), reverse=True)
        unreachable_edge = set()
        print("Edges sorted by number of unreachable edges:")
        print("Total edges checked:", len(self.edge_set))
        for edge in sorted_edges:
            print(edge)
            if edge.get_unreachable_count() > 30:
                unreachable_edge.add(edge.get_id())
        return unreachable_edge
    
    def delete_unreachable_edges(self, output_file):
        shutil.copy(self.route_file, output_file)
        unreachable_edge = self.unreachable_edges_report()
        print(unreachable_edge)
        root = self.tree.getroot()
        for person in root.findall("person"):
            ride = person.find("ride")
            if ride is None:
                # if you later have <walk> persons, handle them here
                continue
            from_edge = ride.get("from")
            to_edge = ride.get("to")
            if from_edge in unreachable_edge or to_edge in unreachable_edge:
                root.remove(person)
        self.tree.write(output_file)
        
        
        
if __name__ == "__main__":
    route_file = "persontripsridecheck.rou.xml"   # change to your file path
    # route_file = "cleaned_routes.rou.xml"   # change to your file path
    net_file = "osm.net.xml"  # change to your file path
    reader = ReqReader(route_file, net_file)
    reader.read_edge()
    reader.check_reachability()
    # reader.delete_unreachable_edges("cleaned_routes.rou.xml")
    reader.unreachable_edges_report()
    
