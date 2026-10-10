"""Graphs from triangle meshes (TINs)."""
import csv

import networkx as nx
import numpy as np
from tqdm import tqdm

from .grid import slope_weighted_length


def mesh_to_graph(edge_filename, vertex_filename):
    f = open(edge_filename)
    all_vertices = []
    lines = f.readlines()
    edges = []
    for i in range(len(lines)):

        vals = lines[i].split()
        edges.append((int(vals[0]), int(vals[1])))
        all_vertices.append(int(vals[0]))
        all_vertices.append(int(vals[1]))
    
    unique_vertices = np.sort(np.unique(all_vertices))

    nx_graph = nx.Graph()

    temp = {}
    for i in range(len(unique_vertices)):
        temp[unique_vertices[i]] = i    
    
    f = open(vertex_filename)

    lines = f.readlines()
    vertices = np.zeros((len(unique_vertices), 3))
    for i in range( len(lines)):
        vals = lines[i].split()
        vertices[i] = [ float(vals[0])/1000, float(vals[1])/1000, float(vals[2])/1000]

    for i in range(len(edges)):
        v1 = temp[edges[i][0]]
        v2 = temp[edges[i][1]]
        weight = np.linalg.norm(vertices[v1] - vertices[v2], ord=2)
        nx_graph.add_edge(v1, v2, weight=weight)

    return nx_graph, vertices

def process_node_features(filename):

    node_idx_dict = {}
    node_feature_lst = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            idx = int(row[0])
            x = float(row[1])/1000
            y = float(row[2])/1000
            z = float(row[3])/1000
            node_idx_dict[idx] = [x, y, z]
            node_feature_lst.append([x, y, z])
            counter += 1
    return node_idx_dict, node_feature_lst

def process_edges(filename):
    edges = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            edges.append([int(row[0]), int(row[1])])
    return edges

def construct_mesh_graph(edges, node_mappings, scale=False):
    G = nx.Graph()
    for edge in tqdm(edges):
        idx1 = int(edge[0])
        idx2 = int(edge[1])
        p1 = np.array(node_mappings[idx1])
        p2 = np.array(node_mappings[idx2])
        w = slope_weighted_length(p1, p2) if scale else np.linalg.norm(p1 - p2, ord=2)
        G.add_edge(idx1, idx2, weight=w)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    return G
