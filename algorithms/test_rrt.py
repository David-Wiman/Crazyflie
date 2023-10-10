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
gnode = np.array([0, 0, 1.7])
options = {
        'N': 20000,
        'terminate_tol': 0.1,
        'npoints': 50,
        'beta': 0.15,
        'lambda': 0.01,
        'r': np.sqrt(0.01),
}

# Run RRT*
path, nodes, parents, costs, gnode_idx = rrt_star.rrt_star(snode, gnode, world, options)

# Evaluate and plot.
print(f'Number of nodes {len(parents)} or {nodes.shape}, number of nodes on path {path.shape}')
rrt_star.plot_path(world, nodes, parents, gnode_idx) # TODO Plot is not working!!!

# plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_xlim([world.xmin, world.xmax])
# ax.set_ylim([world.ymin, world.ymax])
# ax.set_zlim([world.zmin, world.zmax])
# ax.plot3D(*nodes.T, 'o')
# plt.show()