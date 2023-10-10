# Test of RRT* algorithm.

# Imports.
import numpy as np
import rrt_star
from world import BoxWorld, create_mission
import matplotlib.pyplot as plt

# Create world.
world = BoxWorld([[-2, 2], [-1, 1], [0, 2]])
world = create_mission(world, 1)
snode = np.array([0, 0, 0])
gnode = np.array([0, 0, 1.2])
options = {
        'N': 10000,
        'terminate_tol': 0.1,
        'npoints': 50,
        'beta': 0.05,
        'lambda': 0.01,
        'r': np.sqrt(0.4),
}

# Run RRT*
path, nodes, parents, costs = rrt_star.rrt_star(snode, gnode, world, options)

# Evaluate and plot.
print(f'Number of nodes {len(parents)}, number of nodes on path {path.shape}')
rrt_star.plot_path(world, nodes, parents) # TODO Plot is not working!!!