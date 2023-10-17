import numpy as np
import matplotlib.pyplot as plt
from world3D import BoxWorld3D
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
# File for RRT* algorithm.

# Imports.
import numpy as np
import world
import matplotlib.pyplot as plt
from rrt_star import *

def rrt_star_step( gnode, world, options, nodes, parents, costs):
    '''
    Main algorithm for motion planning using RRT*. 
    The algorithm account for static obstacles in the environment.
    All coordinates are in 3D (x,y,z).
    
    Input arguments:
        snode - Starting node for tree with position (x,y,z).
        gnode - Goal end node for tree (x,y,z).
        world - Measurements for the area and list of all static obstacles in the world.
        options - Dictionary with options parameters. 
                {
                'N': <Maximum number of iterations>, 
                'terminate_tol': <Allowed distance to goal node for termination of tree creation>,
                'npoints': <Number of discritization points for checking collisions>,
                'beta': <probability of sampling goal node>,
                'r': <neighborhood radius>,
                }
        
    Output arguments:
        nodes - All node positions in tree.
        parents - Parents to each node in tree. 
        costs - Cost to get to each node in tree.
    '''
    npoints = options.get('npoints')    # Specifies all node positions in network.

    gnode_idx = None

    random_node = sample(world,gnode,options)  # WORLD MEASUREMENTS? ONLY IN FREE SPACE?
    # print(f"nodes: {nodes.shape}")
    nearest_node, nearest_idx = nearest(nodes, random_node)  # Find nearest node.
    new_node = steer(nearest_node, random_node, options) # Steer towards node and create new node. STEP LENGTH?
    # print(f"nearest node: {nearest_node.shape}")
    # print(f"new node: {new_node.shape}")
    # print(discrete_line(nearest_node, new_node, npoints).shape)
    # print(npoints)
    if world.obstacle_free(discrete_line(nearest_node, new_node, npoints)):
        new_idx = len(parents) - 1
        neighbor_idxs = neighborhood(nodes, new_node, options.get('r')) # Find neighborhood to new node. RADIUS OF NEIGHBORHOOD?
        min_cost = distance(nearest_node, new_node) + costs[nearest_idx] # Calculate cost to get to new node from nearest. 
        min_idx = nearest_idx
        
        # For all nodes in neighborhood (connect new to cheapest path).
        for neighbor_idx in neighbor_idxs:  
            neighbor_node = nodes[:,neighbor_idx]
            neighbor_node = neighbor_node[...,np.newaxis]
            # print(f"neighbor node: {neighbor_node.shape}")
            # print(discrete_line(nearest_node, new_node, npoints).shape)
            if world.obstacle_free(discrete_line(neighbor_node, new_node, npoints)) and ( costs[neighbor_idx] + distance(neighbor_node, new_node) < min_cost ): # IF obstacle free AND cost to get to new node is less than before. 
                min_cost = costs[neighbor_idx] + distance(neighbor_node, new_node) # Update nearest and cost.
                min_idx = neighbor_idx
                
        nodes = np.hstack([nodes, new_node]) # Newly added: Nodes are not appending properly.
        parents = np.append(parents, min_idx) #parents.append(min_idx) # Add edge to tree
        costs = np.append(costs, min_cost) # Newly added: Add cost to min.
        
        # For all nodes in neighborhood (rewire tree in neighborhood).
        for neighbor_idx in neighbor_idxs:
            neighbor_node = nodes[:,neighbor_idx]
            
            # IF collision free AND cost to get to neighborhood node is less via new node:

            if world.obstacle_free(discrete_line(nearest_node, new_node, npoints)) and costs[new_idx] + distance(neighbor_node, new_node) < costs[neighbor_idx]:
                parents[neighbor_idx] = new_idx # Set parent for neighborhood node to new node. Remove previous existing edge. Add new edge.

        if distance(new_node, gnode) < options.get('eps'):
            gnode_idx = new_idx
            done = True
    
    # Return tree. 
    print(nodes.shape)
    return nodes, parents, costs, gnode_idx


def sample(world, gnode, options): 
    """Sample a state x in the free state space"""
    
    rg = np.random.default_rng()  # Get the default random number generator
    
    if rg.uniform(0, 1, 1) < options["beta"]:
        gnode = gnode[...,np.newaxis]
        return np.array(gnode)
    else:
        found_random = False
        while not found_random:
            x = rg.uniform(0, 1, 3) * [
                world.xmax - world.xmin,
                world.ymax - world.ymin,
                world.zmax - world.zmin,
            ] + [world.xmin, world.ymin, world.zmin]
            if world.obstacle_free(x[:, None]):
                found_random = True
        return x.reshape(3,1)


def nearest(nodes, random_node):
    """Find index of state nearest to x in the matrix nodes"""
    #nearest_idx = np.argmin(np.sum((nodes - random_node[:, None]) ** 2, axis=0)) 
    # print(f"nodes: {nodes.shape}")
    # print(f"random_node: {random_node.shape}")
    # print(f"size of subtraction: {np.subtract(nodes,random_node).shape}")
    nearest_idx = np.argmin(np.sum(np.subtract(nodes,random_node) ** 2, axis=0)) # TODO test this.
    nearest_node = nodes[:,nearest_idx].reshape(3, 1)
    return nearest_node, nearest_idx


def steer(nearest_node, random_node, options):
    """Steer from nearest_node towards random_node with step size optkions['lambda']

    If the distance to random_node is less than options['lambda'], return
    state random_node.
    """
    dx = np.linalg.norm(random_node - nearest_node)
    if dx < options["lambda"]:
        new_node = random_node
    else:
        new_node = nearest_node + options["lambda"] * (random_node - nearest_node) / dx
    return new_node


def discrete_line(node1, node2, npoints = 50):
    '''Creates a discrete line between two nodes with specified resolution.'''
    return np.linspace(node1, node2, npoints)


def neighborhood(nodes, center, radius):
    """Find the indices of the states in nodes within a neighborhood with
        radius r from node center."""
    idxs = np.where(np.sum((nodes - center) ** 2, axis=0) < radius**2)
    return idxs[0] 


def distance(node1, node2):
    '''Computing distance between two nodes.'''
    return np.linalg.norm(node1 - node2)


def backtrack(parents, nodes):
    '''Backtracks the path from goal node to start node.'''
    idx = parents[-1]
    n_path_nodes = 0
    length = 0

    while idx != 0:
        n_path_nodes +=1
        idx = parents[idx]
        length += np.linalg.norm(nodes[:, parents[idx]] - nodes[:, idx])
    return n_path_nodes, length


def plot_rrt_particle_tree_dynamic_3d(start, goal, world, opts, fig,ax):

    world.draw(ax=ax)
    final_gnode_idx = None
    nodes = np.array([start]).T
    # print(f"nodes.shape: {nodes.shape}")
    lines = []
    start = np.array([[1.0], [1.0], [1.0]])
    nodes = np.array(start)
    parents = np.array([0])  # Specifies edges in network.
    costs = np.array([0])    # Specifies costs to get to node.

    def update(frame):
        #print("test update")
        nonlocal nodes, parents, lines, costs
        # Perform one step of RRT expansion
        nodes, parents, costs, gnode_idx = rrt_star_step(goal, world, opts, nodes, parents, costs)
        # Draw the new lines created in the RRT expansion
        new_lines = []
        for i in range(len(lines), len(nodes[0])):
            parent_idx = parents[i]
            parent_state = nodes[:, parent_idx]
            current_state = nodes[:, i]
            line = ax.plot([parent_state[0], current_state[0]],
                           [parent_state[1], current_state[1]],
                           [parent_state[2], current_state[2]], 'b-', lw=1, alpha=0.5)
            new_lines.extend(line)
        
        if gnode_idx != None:
            final_gnode_idx = gnode_idx
        lines.extend(new_lines)  # Add the new lines to the list of all lines

    ani = FuncAnimation(fig, update, frames=opts['K'], repeat=False)
    return ani, nodes, parents, final_gnode_idx




fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

world = BoxWorld3D([[0, 10], [0, 10], [0, 10]])
# world.add_box(3, 3, 3, 1, 1, 1)  # Define the 3D box with dimensions (width, height, depth)
world.add_box(0, 6, 6, 1, 2, 2)  # Define the 3D box with dimensions (width, height, depth)
world.add_box(6, 1, 5, 4, 2, 2)  # Define the 3D box with dimensions (width, height, depth)
world.add_box(7, 6, 3, 3, 2, 2)  # Define the 3D box with dimensions (width, height, depth)

# Example usage
start = np.array([[1.0], [1.0], [1.0]])
# print(start.shape)
goal = np.array([9.0, 9.0, 9.0])
# Define other parameters, such as world and opts
opts = {'beta': 0.1, 'lambda': 0.1, 'eps': 1, 'K': 1000, 'npoints': 50, 'r':np.sqrt(0.1)}  # Reduced K for demonstration
ax.scatter(start[0], start[1], start[2], c='b', marker='o')
ax.scatter(goal[0], goal[1], goal[2], c='g', marker='*')
# # Plot the RRT tree dynamically in 3D
ani, nodes, parents, gnode_idx = plot_rrt_particle_tree_dynamic_3d(start, goal, world, opts,fig, ax)
ani.save('C:/Users/Simon/OneDrive/Desktop/SkolSaker/TSFS12/tsfs12-crazyflie/algorithms/animation.gif', writer='pillow',savefig_kwargs={'transparent': True, 'bbox_inches': 'tight'})
plt.show()