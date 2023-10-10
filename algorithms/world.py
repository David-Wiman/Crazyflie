"""Class for describing a world with given boundaries and box-shaped
    obstacles."""

"""Taken from the TSFS12 course git, modified to 3D"""

import numpy as np
import matplotlib.pyplot as plt

#num_nodes
#add_box
#draw_box
#register_figure
#draw
#redraw_boxes
#in_bound
#obstacle_free

class BoxWorld:
    def __init__(self, lattice = [[0, 10], [0, 10], [0, 10]]):
        """Create a BoxWorld object with the given lattice"""
        self.st_sp = state_space_from_lattice(lattice)
        self.xmin = np.min(lattice[0])
        self.xmax = np.max(lattice[0])
        self.ymin = np.min(lattice[1])
        self.ymax = np.max(lattice[1])
        self.zmin = np.min(lattice[2])
        self.zmax = np.max(lattice[2])

        self._fig = None
        self._boxes = []
        self.x_obst = np.array([]).reshape((0, 2))
        self.y_obst = np.array([]).reshape((0, 2))
        self.z_obst = np.array([]).reshape((0, 2))

    def num_nodes(self):
        """Get the total number of nodes in the state space"""
        return self.st_sp.shape[1]

    def add_box(self, x, y, z, length, width, height, fill_box=True):
        """ Add a box to the world

        Input
            x - x coordinate of the lower left corner
            y - y coordinate of the lower left corner
            z - z coordinate of the lower left corner
            length - length of the box
            width - width of the box
            height - height of the box
        """
        self._boxes.append((x, y, z, length, width, height, fill_box))
        self.x_obst = np.row_stack((self.x_obst, [x, x + length]))
        self.y_obst = np.row_stack((self.y_obst, [y, y + width]))
        self.z_obst = np.row_stack((self.z_obst, [z, z + height]))

    def draw_box(self, b, *args, ax=None, **kwargs):
        """Help function to function draw, for drawing a box in the figure"""
        x0, y0, z0, W1, W2, W3, fill_box = b

        if fill_box:
            ax.bar3d(x0, y0, z0, W1, W2, W3, shade=True, color='red', alpha=0.7)
        else:
            # Draw the outline of the box
            ax.plot([x0, x0 + W1, x0 + W1, x0, x0],
                    [y0, y0, y0 + W2, y0 + W2, y0], *args, **kwargs)
        '''
        if fill_box:
            print('EHj')
            ax.plot_surface(np.array([x0, x0, x0+W1, x0+W1]), 
                            np.array([y0, y0+W2, y0, y0+W2]),
                            np.zeros(4))
            ax.plot_surface(np.array([-2, -2, -2, -2]), 
                            np.array([y0, y0, y0+W2, y0+W2]),
                            np.array([z0, z0+W3, z0, z0+W3]))
                        #np.array([x0, x0, x0, x0, x0+W1, x0+W1, x0+W1, x0+W1]),
                        #np.array([y0, y0, y0+W2, y0+W2, y0, y0, y0+W2, y0+W2]),
                        #np.append(np.zeros((4)), np.ones(4)))
                        #np.array([z0, z0+W3, z0, z0+W3, z0, z0+W3, z0, z0+W3]))
        else:
            print('då')
            pass
            ax.plot3D([x0, x0, x0, x0, x0+W1, x0+W1, x0+W1, x0+W1],
                    [y0, y0, y0+W2, y0+W2, y0, y0, y0+W2, y0+W2],
                    [z0, z0+W3, z0, z0+W3, z0, z0+W3, z0, z0+W3])
        '''
        
    def register_figure(self, fig):
        self._fig = fig

    def draw(self, *args, ax=None, **kwargs):
        """Draw the obstacles in the world in the figure"""
        if len(args) == 0:
            args = ['r']
        if len(kwargs) == 0:
            kwargs = {'edgecolor': 'k'}
            
        if not ax:
            ax = plt.gca()
                        
        self.redraw_boxes(*args, ax=ax, **kwargs)

    def redraw_boxes(self, *args, ax=None, **kwargs):
        for bi in self._boxes:
            self.draw_box(bi, *args, ax=ax, **kwargs)
        if self._fig:
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

    def in_bound(self, point):
        """Check if a given point is within the world-model boundaries"""
        c = False
        if (point[0] >= self.xmin) and (point[0] <= self.xmax) and \
           (point[1] >= self.ymin) and (point[1] <= self.ymax) and \
           (point[2] >= self.zmin) and (point[2] <= self.zmax):
            c = True
        return c

    def obstacle_free(self, p):
        """ Check if any of a set of points are in collision with obstacles in the world

        Input
          p - numpy array with 2 rows and m columns, where each column represents a point to be checked

        Output
          Returns True if all points are in free space, otherwise False.
        """
        for ii in range(p.shape[1]):
            if obstacle_check(p[0, ii], p[1, ii], p[2, ii], self.x_obst, self.y_obst, self.z_obst):
                return False
        return True


def state_space_from_lattice(lattice):
    """Create a matrix st_sp with all states in the world, given the 
        specified lattice parameters. In the lattice planning, this 3 x N
        matrix is used as a mapping between node number and actual
        coordinates, where the column number is the node number."""
    if len(lattice) == 1:
        st_sp = np.array(lattice[0]).reshape((1, -1))
    else:
        st_sp_1 = state_space_from_lattice(lattice[1:])
        N = st_sp_1.shape[1]
        st_sp = np.array([]).reshape((st_sp_1.shape[0] + 1, 0))
        for xi in lattice[0]:
            st_sp = np.hstack((st_sp,
                               np.row_stack((np.full((1, N), xi),
                                             st_sp_1))))
    return st_sp


def obstacle_check(x, y, z, x_obst, y_obst, z_obst):
    """Help function to function obstacle_free, to check collision for a 
        single point x,y,z"""
    for ii in range(x_obst.shape[0]):
        if (x > x_obst[ii, 0] and x < x_obst[ii, 1]) and \
           (y > y_obst[ii, 0] and y < y_obst[ii, 1]) and \
           (z > z_obst[ii, 0] and z < z_obst[ii, 1]):
            return True
    return False


def create_mission(world, mission_number = 1):
    if mission_number == 1:
        world.add_box(-2, -1, 0, 0.5, 0.5, 0.5)
        world.add_box(-0.5, -0.5, 0.25, 1, 1, 0.3)
    else: 
        pass
    return world