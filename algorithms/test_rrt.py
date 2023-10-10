# Test of RRT* algorithm.

# Imports.
import numpy as np
import rrt_star
from world import BoxWorld
import matplotlib.pyplot as plt

# Create world.
world = BoxWorld([[0, 10], [0, 10], [0, 10]])
snode = np.array([3, 3, 3]).reshape(3, 1)
gnode = np.array([10, 5, 2]).reshape(3, 1)
options = {
        'N': 10000,
        'terminate_tol': 0.1,
        'npoints': 50,
        'beta': 0.05,
        'lambda': 0.3,
        'r': np.sqrt(0.4),
}

# Run RRT*
path, nodes, parents, costs = rrt_star.rrt_star(snode, gnode, world, options)

print(f'Path {path} and shape {path.shape}')

# Evaluate and plot.
rrt_star.plot_path(world, nodes, parents) # TODO Plot is not working!!!
print('Plot done')