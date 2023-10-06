# Test of RRT* algorithm.

# Imports.
import numpy as np
import rrt_star
from world import BoxWorld
import matplotlib as plt

# Create world.
world = BoxWorld([[0, 10], [0, 10], [0, 10]])
snode = np.array([0, 0, 0])
gnode = np.array([10, 5, 2])
options = {
        'N': 10000,
        'terminal_tol': 0.1,
        'npoints': 50,
        'beta': 0.4,
}

# Run RRT*
nodes, parents, costs = rrt_star.rrt_star(snode, gnode, world, options)


# Evaluate and plot.
rrt_star.plot_path()
plt.show()