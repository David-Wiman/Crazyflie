import numpy as np
import matplotlib.pyplot as plt


class BoxWorld3D:
    def __init__(self, lattice):
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
        return self.st_sp.shape[1]

    def add_box(self, x, y, z, width, height, depth, fill_box=True):
        self._boxes.append((x, y, z, width, height, depth, fill_box))
        self.x_obst = np.row_stack((self.x_obst, [x, x + width]))
        self.y_obst = np.row_stack((self.y_obst, [y, y + height]))
        self.z_obst = np.row_stack((self.z_obst, [z, z + depth]))

    def draw_box(self, b, *args, ax=None, **kwargs):
        x0, y0, z0, W1, W2, W3, fill_box = b

        if fill_box:
            ax.bar3d(x0, y0, z0, W1, W2, W3, shade=True, color='blue', alpha=0.7)
        else:
            # Draw the outline of the box
            ax.plot([x0, x0 + W1, x0 + W1, x0, x0],
                    [y0, y0, y0 + W2, y0 + W2, y0], *args, **kwargs)

    def register_figure(self, fig):
        self._fig = fig

    def draw(self, *args, ax=None, **kwargs):
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
        c = False
        if (self.xmin <= point[0] <= self.xmax) and \
           (self.ymin <= point[1] <= self.ymax) and \
           (self.zmin <= point[2] <= self.zmax):
            c = True
        return c

    def obstacle_free(self, p):
        """ Check if any of a set of points are in collision with obstacles in the world

        Input
          p - numpy array with 2 rows and m columns, where each column represents a point to be checked

        Output
          Returns True if all points are in free space, otherwise False.
        """
        # print(f"points to check: {p.shape}")
        for ii in range(p.shape[1]):
            if obstacle_check(p[0, ii], p[1, ii], p[2, ii], self.x_obst, self.y_obst, self.z_obst):
                return False
        return True

def state_space_from_lattice(lattice):
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
    # print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"z: {z}")
    # print(f"x_obst0: {x_obst[0,0]}")
    # print(f"x_obst1: {x_obst[0,1]}")
    for ii in range(x_obst.shape[0]):
        if all(((x > x_obst[ii, 0] and x < x_obst[ii, 1]),
                (y > y_obst[ii, 0] and y < y_obst[ii, 1]),
                (z > z_obst[ii, 0] and z < z_obst[ii, 1]))):
            return True


