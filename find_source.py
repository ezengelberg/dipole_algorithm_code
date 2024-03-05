import argparse
from scipy.spatial import Delaunay
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt




def do(s_filename, ac = True):
    # Get the data file, which is a numpy 2d array.
    # It's made up of: x coordinate, y coordinate, z coordinate, x field, y field, z field.
    # the last three are complex if it's the result of an AC simulation.
    # The first nine lines are preamble.
    if ac:
        aa_field = loadAcMeasurements(s_filename)
    else:
        aa_field = np.loadtxt(s_filename, skiprows = 9)
        
    aa_field = filterMeasurements(aa_field)
    l_roots, graph, tri = rootsAndGraph(aa_field)
    plot(l_roots, graph, tri, aa_field, s_filename)
    # Return the x and y coordinates of the root.
    # If there's more than one root, return the coordinates of the first root.
    # I don't know if this is good enough. I might have to improve it further on.
    return aa_field[np.array(l_roots[0]), 0:2]




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
    # Dividing the magnetic fields by the phase that gives a maximum real part of the z component doesn't give me a nice source of the field.
    # However, it seems that if I see that the z field is generally negative, I can simply flip the direction of the field,
    # and that gives me a nice result.
    # So for now I'll do that.
    if np.sum(np.real(aa_fields[:, 2])) < 0: 
        aa_fields = -aa_fields # For debugging.
    aa_fields = np.real(aa_fields)
    aa_field = np.hstack((aa_locations, aa_fields))
    return aa_field




def plot(l_roots, graph, tri, aa_field, s_filename):
    fig = plt.figure()
    # Plot the Delaunay grid.
    plt.triplot(aa_field[:,0], aa_field[:,1], tri.simplices)
    # Print the resulting roots and some details.
    print('Results:')
    print(l_roots)
    print([len(nx.descendants(graph, root)) for root in l_roots])
    print([aa_field[root] for root in l_roots])
    # Plot the vertices and the roots.
    plt.plot(aa_field[:,0], aa_field[:,1], 'o', linestyle = 'None', markersize = 1)
    a_roots = np.array(l_roots)
    plt.plot(aa_field[a_roots,0], aa_field[a_roots,1], 'o', linestyle = 'None', color = 'red', markersize = 10)
    for edge in graph.edges():
        # Plot the edges as arrows.
        plt.arrow(aa_field[edge[0], 0], aa_field[edge[0], 1],
                  aa_field[edge[1], 0] - aa_field[edge[0], 0], aa_field[edge[1], 1] - aa_field[edge[0], 1],
                  head_width = 200, head_length = 400, fc='green', ec='green', length_includes_head = True)
        # Plot the magnetic fields on the planes as arrows.
        plt.arrow(aa_field[edge[0], 0], aa_field[edge[0], 1],
                  aa_field[edge[0], 3], aa_field[edge[0], 4],
                  head_width = 500, head_length = 1000, fc='black', ec='black')
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
        a_field_vector = aa_field[vertex, 3:5]
        a_field_direction = a_field_vector / np.linalg.norm(a_field_vector)
        #print(a_field_direction)
        # Find the directions from the neighboring vertices to the vertex.
        a_vertex_neighbors = a_neighbors[a_starts[vertex] : a_starts[vertex + 1]]
        a_point = aa_field[vertex, 0:2]
        aa_neighboring_points = aa_field[a_vertex_neighbors, 0:2]
        aa_vectors_from_neighbors_to_point = a_point - aa_neighboring_points
        aa_directions_from_neighbors_to_point = \
            aa_vectors_from_neighbors_to_point / np.linalg.norm(aa_vectors_from_neighbors_to_point, axis = 1)[:, np.newaxis]
        #print(aa_directions_from_neighbors_to_point)
        a_alignment = np.dot(aa_directions_from_neighbors_to_point, a_field_direction)
        #print(a_alignment)
        parent_index_in_list_of_neighbors = np.argmax(a_alignment)
        parent = a_vertex_neighbors[parent_index_in_list_of_neighbors]
        # Add an edge from the parent to the vertex.
        g.add_edge(parent, vertex)
        #print(g)
        
    # Remove the cycles in the graph.
    print('Removing cycles.')
    while True:
        # Get the first cycle.
        ll_cycles = nx.simple_cycles(g)
        try:
            l_cycle = next(ll_cycles)
            print(l_cycle)
        # If there's no first cycle, we're done.
        except StopIteration:
            break
        # In each cycle, we assume that the node with the largest z component of the field should not have a parent.
        # I saw that this isn't always optimal. But it's good enough for me for now.
        print(aa_field[np.array(l_cycle)])
        a_z_field_components = aa_field[np.array(l_cycle), 5]
        index_in_cycle_with_largest_z_field = np.argmax(a_z_field_components)
        node_with_largest_z_field = l_cycle[index_in_cycle_with_largest_z_field]
        print(node_with_largest_z_field)
        parent_of_node_with_largest_z_field = l_cycle[index_in_cycle_with_largest_z_field - 1]
        g.remove_edge(parent_of_node_with_largest_z_field, node_with_largest_z_field)
                
    # Find the roots of the trees in the graph, now that it doesn't have cycles.
    l_roots = [node for node, in_degree in g.in_degree() if in_degree == 0]
    # Return the root of the biggest tree in the graph.
    return l_roots, g, tri
    


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help = 'Folder in which to find the input file, named "ground_level.txt", and to create the outputs.')
    parser.add_argument('--ac', action = 'store_true', help = 'Use this if the input is an AC field (complex numbers).')
    args = parser.parse_args()
    os.chdir(args.folder)
    do('ground_level.txt', args.ac)
