import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from cmp_geometry.vect_handler import VectorHandler
from cmp_geometry.handlers import vect_equal

def in_poly(point, poly, r):
    ''' Checks whether point (within error r) is in the set of points poly

    input: point - numpy array of a single point
           poly - set of numpy arrays that form a polygon
           r - error or uncertainty in x and y values of point

    returns: True if point is in poly within error r, False otherwise
    '''
    s_poly = Polygon(poly)
    p = Point(point[0], point[1])
    c = p.buffer(r)
    s_poly = Polygon(poly)
    if c.intersects(s_poly):
        in_p = True
    else:
        in_p = False
    return in_p


def arrange_vects(vects):
    ''' Arrange the vectors in the list vects based on counterclockwise angle from +x axis
    '''
    x = np.array([1, 0])
    x_h = VectorHandler(x)
    # Get angles from x axis counterclockwise
    angles = [x_h.get_pos_angle(vect) for vect in vects]

    # Sort vectors based on angles
    zipped = list(zip(vects, angles))
    zipped.sort(key = lambda x: x[1])

    # Take only the vectors after sorting
    arranged = [tup[0] for tup in zipped]
    return arranged

def shifted(point, corners, center_basis, r):
    if in_poly(point, corners, r):
        return point
    elif in_poly(np.subtract(point, center_basis[0]), corners, r):
        return np.subtract(point, center_basis[0])
    elif in_poly(np.subtract(point, center_basis[1]), corners, r):
        return np.subtract(point, center_basis[1])
    else:
        return np.subtract(point, np.add(center_basis[0], center_basis[1]))

def get_index(point, pos, r):
    count = 0
    for vec in pos:
        if vect_equal(point, vec, r):
            return count
        count += 1
    
    if count == 0:
        raise Exception("ERROR: Could not match a vector to an index")


class LatticeStructure():

    def __init__(self, group, cutoff, cutoff_type='m', size=1, error=10**(-8)):
        self.size = size
        self.cutoff = cutoff
        self.group = group

        # Set basis given lattice type
        if group.lower() in ['h', 'hex', 'hexagon', 'hexagonal', 'honeycomb']:
            self.type = 'h'
            self.basis = [np.array([1, 0]), np.array([1/2, np.sqrt(3)/2])]
            self.centers = False
        elif group.lower() in ['t', 'tri', 'triangular', 'triangle']:
            self.type = 't'
            self.basis = [np.array([1, 0]), np.array([1/2, np.sqrt(3)/2])]
            self.centers = True
        else: # Assume square
            self.type = 's'
            self.basis = [np.array([1, 0]), np.array([0, 1])]
            self.centers = True

        # Configure cutoff type
        if cutoff_type.lower() in ['m', 'mag', 'magnitude']:
            self.cutoff_type = 'm'
        else:
            self.cutoff_type = 'n'
            if not isinstance(self.cutoff, int):
                raise Exception("ERROR: Non-integer cutoff.")

        self.basis = [base*size for base in self.basis]
        self.error = error
    
    def pos(self):
        pos = list()

        # Generate lattice with centers and calculate centers
        if self.cutoff_type == 'm':
            max_int = 2*int(self.cutoff/self.size)

            for i in range(-max_int, max_int+1):
                for j in range(-max_int, max_int+1):
                    # Remove centers if requested
                    if (not self.centers) and (i - j) % 3 == 0 and (2*i +j) % 3 == 0: continue

                    # Candidate position
                    cand = i*self.basis[0] + j*self.basis[1]
                    if np.linalg.norm(cand) >= self.cutoff: continue

                    # Add position not already existing
                    exists_cand = [abs(cand[0] - a[0]) < self.error and abs(cand[1] - a[1]) < self.error for a in pos]
                    if not np.any(exists_cand):
                        pos.append(cand)
        else:
            max_int = int(self.cutoff/3+5)

            for i in range(-max_int, max_int+1):
                for j in range(-max_int, max_int+1):
                    # Remove centers if requested
                    if (not self.centers) and (i - j) % 3 == 0 and (2*i +j) % 3 == 0: continue

                    # Candidate position
                    cand = i*self.basis[0] + j*self.basis[1]
                    # Add position not already existing
                    exists_cand = [abs(cand[0] - a[1][0]) < self.error and abs(cand[1] - a[1][1]) < self.error for a in pos]
                    if not np.any(exists_cand):
                        pos.append((np.linalg.norm(cand), cand))
    
            rounding = int(-1*np.log10(self.error)-1)

            # Regroup elements
            grouped_list = [(k, list(g)) for k, g in groupby(pos, lambda tup: round(tup[0], rounding))]

            # Sort and reformat
            grouped_list.sort(key=lambda tup:tup[0])
            grouped_list_ordered = list()
            for i in range(self.cutoff+1):
                grouped_list_ordered.append([grouped_list[i][0], list()])
                for j in range(len(grouped_list[i][1])):
                    grouped_list_ordered[i][1].append(grouped_list[i][1][j][1])

            combined = list()
            for i in range(len(grouped_list_ordered)):
                combined.extend(grouped_list_ordered[i][1])
            
            # Only take up to cutoff
            cut_vectors = list()
            for i in range(len(combined)):
                cut_vectors.append(combined[i])
                if self.cutoff > len(cut_vectors):
                    continue
                else:
                    break

            pos = cut_vectors
        return pos

    def neighbors(self, point, pos):
        # Determine possible shifts to neighbors
        shift_set = list()
        for i, j in range(-1, 1):
            if i == 0 and j == 0: continue
            cand = i*self.basis[0] + j*self.basis[1]
            if np.linalg.norm(cand) <= 1:
                shift_set.append(cand)

        # Get neighbors
        neigh = list()
        is_in = [abs(cand[0] - a[0]) < self.error and abs(cand[1] - a[1]) < self.error for a in pos]
        for vec in shift_set:
            cand = point + vec
            if np.any(is_in):
                neigh.append(cand)

        return neigh

    def graph(self):
        if self.cutoff_type != 'm':
            raise Exception("ERROR: Not yet implemented cutoff_type")

        max_int = int(self.cutoff/self.size)
        pos = list()
        graph = list()

        # Determine index shift possbilities
        shift_set = list()
        for i in range(-1, 1):
            for j in range(-1, 1):
                if i == 0 and j == 0: continue
                cand = i*self.basis[0] + j*self.basis[1]
                if np.linalg.norm(cand) <= 1:
                    shift_set.append((i, j))

        # Generate lattice with centers and calculate centers
        for i in range(-max_int, max_int+1):
            for j in range(-max_int, max_int+1):
                # Remove centers if requested
                if (not self.centers) and (i - j) % 3 == 0 and (2*i +j) % 3 == 0: continue

                # Candidate position; ensure within cutoff
                cand = i*self.basis[0] + j*self.basis[1]
                if np.linalg.norm(cand) >= self.cutoff: continue

                # Retrieve neighbors
                neighbors = list()
                for ind in shift_set:
                    n = ind[0]
                    m = ind[1]
                    neigh_cand = (i+n)*self.basis[0] + (j+m)*self.basis[1]
                    
                    # Ensure not center or in cut off

                    if self.centers:
                        neighbors.append(neigh_cand)
                    else:
                        if not ((i + n - j - m) % 3 == 0 and (2*i+ 2*n +j + m) % 3 == 0):
                            neighbors.append(neigh_cand)

                # Add position not already existing
                exists_cand = [abs(cand[0] - a[0]) < self.error and abs(cand[1] - a[1]) < self.error for a in pos]
                if not np.any(exists_cand):
                    pos.append(cand)
                    graph.append((cand, neighbors))

        return graph

    def generate_shells(self, n_shells):
        max_int = n_shells
        pos = list()
        mag_tup_list = list()

        rounding = int(-1*np.log10(self.error)-1)

        # Generate lattice with centers and calculate centers
        for i in range(-max_int, max_int+1):
            for j in range(-max_int, max_int+1):
                if i == 0 and j == 0: continue
                # Remove centers if requested
                if (not self.centers) and (i - j) % 3 == 0 and (2*i +j) % 3 == 0: continue

                # Candidate position; ensure within cutoff
                cand = i*self.basis[0] + j*self.basis[1]

                # Add position not already existing
                exists_cand = [abs(cand[0] - a[0]) < self.error and abs(cand[1] - a[1]) < self.error for a in pos]
                if not np.any(exists_cand):
                    pos.append(cand)
                    mag_tup_list.append((np.linalg.norm(cand), cand))
        
        mag_tup_list.sort(key=lambda tup: tup[0])

        # Regroup elements
        shell_list = [(k, list(g)) for k, g in groupby(mag_tup_list, lambda tup: round(tup[0], rounding))]

        # Sort and reformat
        shell_list.sort(key=lambda tup:tup[0])

        shell_list_ordered = list()
        for i in range(n_shells):
            shell_list_ordered.append([shell_list[i][0], list()])
            for j in range(len(shell_list[i][1])):
                shell_list_ordered[i][1].append(shell_list[i][1][j][1])
        return shell_list_ordered

    def periodic_lattice(self):
        max_int = int(self.cutoff/self.size)
        mesh_basis = [self.basis[0] + self.basis[1], -1*self.basis[0] + 2*self.basis[1]]
        center_basis = [max_int*base for base in mesh_basis]

        r = np.linalg.norm(self.basis[0])*(1/max_int)*(1/10)
        # Determine index shift possbilities
        shift_set = list()
        bound_corners = list()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                cand = i*self.basis[0] + j*self.basis[1]
                if np.linalg.norm(cand) <= self.size:
                    shift_set.append((i, j))
                    bound_corners.append(cand*max_int)
        plt.scatter([corner[0] for corner in bound_corners], [corner[1] for corner in bound_corners])

        corners = arrange_vects(bound_corners)

        mesh = list()
        indices = list()
        for i in range(max_int):
            for j in range(max_int):
                # Remove centers if requested
                if (not self.centers) and (i - j) % 3 == 0 and (2*i +j) % 3 == 0: continue

                # Add to mesh if in polygon
                point = np.add(mesh_basis[0]*i, mesh_basis[1]*j)
                
                if in_poly(point, corners, r):
                    upd_point = point
                elif in_poly(np.subtract(point, center_basis[0]), corners, r):
                    upd_point = np.subtract(point, center_basis[0])
                elif in_poly(np.subtract(point, center_basis[1]), corners, r):
                    upd_point = np.subtract(point, center_basis[1])
                else:
                    upd_point = np.subtract(point, np.add(center_basis[0], center_basis[1]))
                mesh.append(upd_point)
                indices.append([i, j])

        return mesh, indices

    def periodic_graph(self):
        max_int = int(self.cutoff/self.size)
        mesh_basis = [self.basis[0] + self.basis[1], -1*self.basis[0] + 2*self.basis[1]]
        center_basis = [max_int*base for base in mesh_basis]

        r = np.linalg.norm(self.basis[0])*(1/max_int)*(1/10)
        # Determine index shift possbilities
        shift_set = list()
        bound_corners = list()
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                cand = i*self.basis[0] + j*self.basis[1]
                if np.linalg.norm(cand) <= self.size:
                    shift_set.append((i, j))
                    bound_corners.append(cand*max_int)

        corners = arrange_vects(bound_corners)

        pos, indices =  self.periodic_lattice()
        graph = list()
        count = 0
        for p in pos:
            i, j = indices[count]

            # Retrieve neighbors
            neighbors = list()
            for ind in shift_set:
                n = ind[0]
                m = ind[1]
                neigh_cand = (i+n)*mesh_basis[0] + (j+m)*mesh_basis[1]
                
                if self.centers or not ((i + n - j - m) % 3 == 0 and (2*i+ 2*n +j + m) % 3 == 0):
                    shifted_neigh = shifted(neigh_cand, corners, center_basis, r)
                    neighbors.append((get_index(shifted_neigh, pos, r), shifted_neigh))
                
            graph.append((p, neighbors))
            
            count += 1

        return graph

    def plot_lattice(self):
        pos = self.pos()
        length = len(pos)
        x = [pos[i][0] for i in range(length)]
        y = [pos[i][1] for i in range(length)]
        plt.scatter(x, y)
        plt.show

    def plot_shells(self, n_shells):
        shells_raw = self.generate_shells(n_shells)
        shells = [shell[1] for shell in shells_raw]

        length = len(shells)
        color_base = ['k', 'b']

        count = 0
        for shell_set in shells:
            length = len(shell_set)
            x = [shell_set[i][0] for i in range(length)]
            y = [shell_set[i][1] for i in range(length)]
            plt.scatter(x, y, c=color_base[count % 2])
            count += 1
        plt.show

    def plot_lattice_periodic(self):
        pos, _ = self.periodic_lattice()
        length = len(pos)
        x = [pos[i][0] for i in range(length)]
        y = [pos[i][1] for i in range(length)]
        plt.scatter(x, y, color='b')
        plt.show
    



if __name__ == "__main__":
    size = 1
    cutoff = 6
    lattice = LatticeStructure('hex', cutoff=cutoff, size=size, cutoff_type='m')
    lattice.plot_lattice_periodic()
    a = lattice.periodic_graph()
    print(a[7], a[11][0])
    #lattice.plot_shells(2)