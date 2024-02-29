import argparse
from scipy.spatial import Delaunay
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt




def complexConverter(s):
    return complex(s.replace('i', 'j'))




def do(aa_field):

    # Do the Delaunay tesselation, using the x and y coordinates of the points.
    aa_points = aa_field[:, 0:2]
    tri = Delaunay(aa_points)
    # Plot this.
    plt.triplot(aa_points[:,0], aa_points[:,1], tri.simplices)
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
    return l_roots, g
    


    
if __name__ == '__main__':

    # Get the data file, which is a numpy 2d array.
    # It's made up of: x coordinate, y coordinate, z coordinate, x field, y field, z field.
    # the last three are complex.
    # The first nine lines are preamble.
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help = 'Folder in which to find the input file, named "ground_level.txt", and to create the outputs.')
    args = parser.parse_args()
    os.chdir(args.folder)
    # Replace all i's with j's in the data file, because that's what python needs when it reads complex numbers.
    with open('ground_level.txt', 'r') as file:
        file_content = file.read()
        file_content_replaced = file_content.replace('i', 'j')
    # Write the modified content back to the file.
    with open('ground_level_modified.txt', 'w') as file:
        file.write(file_content_replaced)
    print("Replacement completed. Modified content saved to 'ground_level_modified.txt'.")
    # Now load the modified file, with j's instead of i's, just the way python likes it.
    aa_field = np.loadtxt('ground_level_modified.txt', skiprows = 9, dtype = complex)
    aa_field = np.real(aa_field)
    
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

    fig = plt.figure()
    # Call the function that finds the root.
    l_roots, g = do(aa_field)
    print('Results:')
    print(l_roots)
    print([len(nx.descendants(g, root)) for root in l_roots])
    print([aa_field[root] for root in l_roots])
    # Plot the vertices and the roots.
    plt.plot(aa_field[:,0], aa_field[:,1], 'o', linestyle = 'None', markersize = 1)
    a_roots = np.array(l_roots)
    plt.plot(aa_field[a_roots,0], aa_field[a_roots,1], 'o', linestyle = 'None', color = 'red', markersize = 10)
    # Plot the edges as arrows.
    for edge in g.edges():
        plt.arrow(aa_field[edge[0], 0], aa_field[edge[0], 1],
                  aa_field[edge[1], 0] - aa_field[edge[0], 0], aa_field[edge[1], 1] - aa_field[edge[0], 1],
                  head_width = 500, head_length = 1000, fc='green', ec='green')
        plt.arrow(aa_field[edge[0], 0], aa_field[edge[0], 1],
                  aa_field[edge[0], 3], aa_field[edge[0], 4],
                  head_width = 500, head_length = 1000, fc='black', ec='black')
    plt.show()
    fig.savefig('map.png')
    plt.close()

