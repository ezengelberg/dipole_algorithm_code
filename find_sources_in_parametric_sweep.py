import argparse
import find_source
import numpy as np
import os
import matplotlib.pyplot as plt
import re




class Dipole:

    def __init__(self, s_filename, noise):
        # Extract the numbers for x and y in each of the files.
        ls_filename = s_filename.split('.')[0].split('_')
        self.a_known_location = np.array((int(ls_filename[4]), int(ls_filename[7])))
        # Get the root coordinates for x and y in each of the files.
        self.a_derived_location = find_source.do(s_filename, ac = True, noise = noise)


    def plot(self):
        plt.plot(*self.a_derived_location, 'o', linestyle = 'None', markersize = 1, color = 'red')
        plt.plot(*self.a_known_location, 'x', linestyle = 'None', markersize = 1, color = 'blue')
        plt.arrow(*self.a_known_location, *(self.a_derived_location - self.a_known_location), head_width = 200,
                  head_length = 400, fc='black', ec='black', length_includes_head = True)

        

        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help = 'Folder in which to find the input file, named "ground_level.txt", and to create the outputs.')
    parser.add_argument('--noise', type = float, default = 0.0, help = 'Noise level to add to the measurements, in Tesla.')
    args = parser.parse_args()
    os.chdir(args.folder)
    # Get all the files of the form "ground_level_dipole_x_<number>_dipole_y_<number>".
    ls_files = [s_filename for s_filename in os.listdir() if re.fullmatch(r'ground_level_dipole_x_\d+_dipole_y_\d+\.txt', s_filename)]
    # Caclulate the known and derived locations of the dipoles.
    l_dipoles = [Dipole(s_filename, noise = args.noise) for s_filename in ls_files]
    # Plot the known and derived locations of the dipoles.
    fig = plt.figure()
    for dipole in l_dipoles:
        dipole.plot()
    plt.show()
    fig.savefig('map.png')
    plt.close()

