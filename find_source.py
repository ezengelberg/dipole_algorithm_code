import argparse
from scipy.spatial import Delaunay
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt




# This function decides which phase makes more sense. It can return either the string 'original' or the string 'opposite'.
def correctPhase(l_roots, graph, l_opposite_roots, opposite_graph):
    # If one of the phases doesn't have a root, choose the other.
    if not l_opposite_roots:
        return 'original'
    if not l_roots:
        return 'opposite'
    # If both phases have roots, choose the phase with the root that has the most descendants.
    a_descendants_original = np.array([len(nx.descendants(graph, root)) for root in l_roots])
    a_descendants_opposite = np.array([len(nx.descendants(opposite_graph, root)) for root in l_opposite_roots])
    if np.max(a_descendants_original) > np.max(a_descendants_opposite):
        return 'original'
    else:
        return 'opposite'




def do(s_filename, ac = True, noise = 0):
    print(f'\nExecuting algorithm for {s_filename}.')
    # Get the data file, which is a numpy 2d array.
    # It's made up of: x coordinate, y coordinate, z coordinate, x field, y field, z field.
    # the last three are complex if it's the result of an AC simulation.
    # The first nine lines are preamble.
    if ac:
        aa_field = loadAcMeasurements(s_filename)
    else:
        aa_field = np.loadtxt(s_filename, skiprows = 9)
    # If instructed, add noise.
    if noise:
        rng = np.random.default_rng()
        aa_noise = noise * rng.standard_normal(size = (len(aa_field), 3))
        aa_field[:, 3:6] += aa_noise
        # Save the array with the noise added, for future reference.
        s_noisy_filename = s_filename.split('.')[0] + f'_noise_level_{noise}.txt'
        np.savetxt(s_noisy_filename, aa_field)
        
    aa_field = filterMeasurements(aa_field)
    l_roots, graph, tri = rootsAndGraph(aa_field)
    # Dirty trick: In the AC case, I don't know if the field is flowing in or out.
    # So I just do the calculation for both possibilities, and then take the solution which makes more sense.
    if ac:
        aa_opposite_field = np.hstack((aa_field[:, 0:3], -aa_field[:, 3:6]))
        l_opposite_roots, opposite_graph, opposite_tri = rootsAndGraph(aa_opposite_field)
        if correctPhase(l_roots, graph, l_opposite_roots, opposite_graph) == 'opposite':
            aa_field = aa_opposite_field
            l_roots = l_opposite_roots
            graph = opposite_graph
            tri = opposite_tri
    # Return the coordinates of the root.
    # If there's more than one root, return the coordinates of the root with the most descendants.
    if len(l_roots) > 1:
        print('I got more than one root.')
    a_descendants = np.array([len(nx.descendants(graph, root)) for root in l_roots])
    print(f'The numbers of descendants are {a_descendants}.')
    real_root = l_roots[np.argmax(a_descendants)]
    # Plot.
    plot(real_root, l_roots, graph, tri, aa_field, s_filename)
    # Return the x and y coordinates of the root.
    return aa_field[real_root, 0:2]




def filterMeasurements(aa_field):
    # At the points at the edges of the grid, for some reason, Comsol shows an anomalous direction of the field.
    # That causes the script to identify roots where there are none really.
    # To solve this, I'm getting rid of all points where the magnitude of the magnetic field is less than 1/10 of the maximum.
    # I think that's a good idea to do anyway, because it can help with noise.
    a_field_magnitudes = np.linalg.norm(aa_field[:, 3:6], axis = 1)
    field_magnitude_max = max(a_field_magnitudes)
    print(field_magnitude_max)
    np.savetxt('field_magnitudes.txt', a_field_magnitudes)
    print(f'{len(aa_field)} points measured.')
    aa_field = aa_field[a_field_magnitudes > field_magnitude_max / 10]
    print(f'{len(aa_field)} points where the field magnitude is greater than 1/10 of the maximum.')
    return aa_field




def getBoundaryNodes(tri):
    ll_edges = []
    for l_simplex in tri.simplices:
        ll_simplex_edges = [sorted((l_simplex[i - 1], l_simplex[i])) for i in range(3)]
        ll_edges += ll_simplex_edges
    a_all_nodes = np.unique(np.array(ll_edges).flatten())
    print(f'There are {len(a_all_nodes)} nodes.')
    ll_boundary_edges = [l_edge for l_edge in ll_edges if ll_edges.count(l_edge) == 1]
    a_boundary_nodes = np.unique(np.array(ll_boundary_edges).flatten())
    print(f'There are {len(a_boundary_nodes)} boundary nodes.')
    return a_boundary_nodes
            



def loadAcMeasurements(s_filename):
    # Replace all i's with j's in the data file, because that's what python needs when it reads complex numbers.
    with open(s_filename, 'r') as file:
        file_content = file.read()
        file_content_replaced = file_content.replace('i', 'j')
    # Write the modified content back to the file.
    s_modified_filename = s_filename.split('.')[0] + '_modified.txt'
    with open(s_modified_filename, 'w') as file:
        file.write(file_content_replaced)
    # Now load the modified file, with j's instead of i's, just the way python likes it.
    # Load the locations as real numbers.
    aa_locations = np.loadtxt(s_modified_filename, skiprows = 9, usecols = (0, 1, 2))
    # Load the magnetic fields as complex numbers.
    aa_fields = np.loadtxt(s_modified_filename, skiprows = 9, usecols = (3, 4, 5), dtype = complex)
    aa_fields = np.real(aa_fields)
    aa_field = np.hstack((aa_locations, aa_fields))
    return aa_field




def plot(real_root, l_roots, graph, tri, aa_field, s_filename):
    fig = plt.figure()
    # Plot the Delaunay grid.
    plt.triplot(aa_field[:,0], aa_field[:,1], tri.simplices)
    # Plot the real roots.
    plt.plot(aa_field[real_root, 0], aa_field[real_root, 1], 'o', linestyle = 'None', color = 'red', markersize = 10)
    # Plot other roots, if they exist.
    for root in l_roots:
        plt.plot(aa_field[root, 0], aa_field[root, 1], 'o', linestyle = 'None', color = 'red', markersize = 5)
    # Plot the edges as arrows.
    for edge in graph.edges():
        plt.arrow(aa_field[edge[0], 0], aa_field[edge[0], 1],
                  aa_field[edge[1], 0] - aa_field[edge[0], 0], aa_field[edge[1], 1] - aa_field[edge[0], 1],
                  head_width = 20, head_length = 40, fc='green', ec='green', length_includes_head = True)
    # Plot the magnetic fields on the planes as arrows.
    # But only do it for points which were used in the Delaunay grid.
    a_coplanar = tri.coplanar[:, 0]
    a_boolean = np.ones(aa_field.shape[0], dtype = bool)
    a_boolean[a_coplanar] = False
    aa_filtered_field = aa_field[a_boolean]
    for a_point in aa_filtered_field:
        plt.arrow(a_point[0], a_point[1], a_point[3], a_point[4], head_width = 20, head_length = 40, fc='black', ec='black')
    #plt.show()
    fig.savefig(s_filename.split('.')[0] + '_map.png')
    plt.close()




def rootsAndGraph(aa_field):

    # Do the Delaunay tesselation, using the x and y coordinates of the points.
    aa_points = aa_field[:, 0:2]
    tri = Delaunay(aa_points)
    # Get the neighbors of each vertex.
    a_starts, a_neighbors = tri.vertex_neighbor_vertices
    np.savetxt('delaunay_tesselation_starts.txt', a_starts, fmt = '%d')
    np.savetxt('delaunay_tesselation_neighbors.txt', a_neighbors, fmt = '%d')
    np.savetxt('simplices.txt', tri.simplices, fmt = '%d')
    np.savetxt('coplanar.txt', tri.coplanar, fmt = '%d')
    # Prepare a directed graph, initially empty, to add the edges to.
    g = nx.DiGraph()

    # For each vertex, find its parent vertex, i.e. the one which the field came from. 
    for vertex in range(len(a_starts) - 1):
        # Check if the vertex have neighbors. Not all vetrices are used in the Delaunay tesselation.
        if a_starts[vertex] == a_starts[vertex + 1]:
            continue
        # Find the normalized field in x and y at the vertex.
        a_field = aa_field[vertex, 3:5]
        # Find the directions from the neighboring vertices to the vertex.
        a_vertex_neighbors = a_neighbors[a_starts[vertex] : a_starts[vertex + 1]]
        a_point = aa_field[vertex, 0:2]
        aa_neighboring_points = aa_field[a_vertex_neighbors, 0:2]
        aa_vectors_from_neighbors_to_point = a_point - aa_neighboring_points
        aa_directions_from_neighbors_to_point = \
            aa_vectors_from_neighbors_to_point / np.linalg.norm(aa_vectors_from_neighbors_to_point, axis = 1)[:, np.newaxis]
        #print(aa_directions_from_neighbors_to_point)
        a_scalar_projection = np.dot(aa_directions_from_neighbors_to_point, a_field)
        #print(a_alignment)
        parent_index_in_list_of_neighbors = np.argmax(a_scalar_projection)
        parent = a_vertex_neighbors[parent_index_in_list_of_neighbors]
        # Add an edge from the parent to the vertex.
        # The weight of the edge is the scalar projection of the field on the direction from the neighbor to the node.
        g.add_edge(parent, vertex, weight = a_scalar_projection[parent_index_in_list_of_neighbors])
        
    # Remove the cycles in the graph.
    print('Removing cycles.')
    while True:
        # Get the first cycle.
        ll_cycles = nx.simple_cycles(g)
        try:
            l_cycle = next(ll_cycles)
            print(f'Found a cycle: {l_cycle}.')
        # If there's no first cycle, we're done.
        except StopIteration:
            break
        # If there is a cycle, we remove the edge with the least weight.
        l_edges_in_cycle = [(l_cycle[i - 1], l_cycle[i]) for i in range(len(l_cycle))]
        a_weights = np.array([g[edge[0]][edge[1]]['weight'] for edge in l_edges_in_cycle])
        edge_to_remove = l_edges_in_cycle[np.argmin(a_weights)]
        g.remove_edge(edge_to_remove[0], edge_to_remove[1])
        
    # Find the roots of the trees in the graph, now that it doesn't have cycles.
    l_roots = np.array([node for node, in_degree in g.in_degree() if in_degree == 0])
    # Remove roots which are at the outer boundaries of the tesselation.
    a_boundary_nodes = getBoundaryNodes(tri)
    l_roots = [node for node in l_roots if node not in a_boundary_nodes]
    # Return the root of the biggest tree in the graph.
    return l_roots, g, tri
    


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help = 'Folder in which to find the input file, named "ground_level.txt", and to create the outputs.')
    parser.add_argument('--ac', action = 'store_true', help = 'Use this if the input is an AC field (complex numbers).')
    parser.add_argument('--noise', type = float, default = 0.0, help = 'Noise level to add to the measurements, in Tesla.')
    args = parser.parse_args()
    os.chdir(args.folder)
    do('ground_level.txt', args.ac, args.noise)
