# File for RRT* algorithm.

# Imports.
import numpy as np
import world
import matplotlib.pyplot as plt

def rrt_star(snode, gnode, world, options):
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
    
    N = options.get('N')
    npoints = options.get('npoints')
    
    nodes = np.array(snode)    # Specifies all node positions in network.
    parents = np.array([None])  # Specifies edges in network.
    costs = np.array([0])    # Specifies costs to get to node.
    
    for i in range(N):
        random_node = sample(world,gnode,options)  # WORLD MEASUREMENTS? ONLY IN FREE SPACE?
        nearest_node, nearest_idx = nearest(nodes, random_node)  # Find nearest node.
        new_node = steer(nearest_node, random_node, options) # Steer towards node and create new node. STEP LENGTH?
        
        if world.obstacle_free(discrete_line(nearest_node, new_node, npoints)):
            new_idx = len(parents) - 1
            neighbor_idxs = neighborhood(nodes, new_node, options.get('r')) # Find neighborhood to new node. RADIUS OF NEIGHBORHOOD?
            min_cost = distance(nearest_node, new_node) + costs[nearest_idx] # Calculate cost to get to new node from nearest. 
            min_idx = nearest_idx
            
            # For all nodes in neighborhood (connect new to cheapest path).
            for neighbor_idx in neighbor_idxs:  
                neighbor_node = nodes[:,neighbor_idx]
                
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
    
            # Checking if new node is the goal node.
            if distance(new_node, gnode) < options.get('terminate_tol'):
                break
    
    # Return tree. 
    return nodes, parents, costs


def sample(world, gnode, options): 
    """Sample a state x in the free state space"""
    
    rg = np.random.default_rng()  # Get the default random number generator
    
    if rg.uniform(0, 1, 1) < options["beta"]:
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
    nearest_idx = np.argmin(np.sum((nodes - random_node) ** 2, axis=0)) # TODO test this.
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
    return idxs[0] # NOTE double index?


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


def plot_path(world, nodes, parents): # Needs modifications for 3D.
    '''Plots the tree and the planned path.'''
    
    fig = plt.figure()
    #world.draw()
    ax = plt.axes(projection='3d')
    
    # Plot tree
    drawlines = []
    # for node in parents:
    #     if node != 0:
    #         ll = np.column_stack((nodes[:, parents[node]], nodes[:, node]))
    #         drawlines.append(ll[0])
    #         drawlines.append(ll[1])
    #         drawlines.append(ll[2])
    # #plt.plot(*drawlines, color='k', lw=1)
    # ax.plot3D(*drawlines, color='k', lw = 1)
     
    drawlines = []
    idx = -1 # idx_goal before.
    # Plot path.
    while idx != 0:
        ll = np.column_stack((nodes[:, parents[idx]], nodes[:, idx]))
        drawlines.append(ll[0])
        drawlines.append(ll[1])
        drawlines.append(ll[2])
        idx = parents[idx]
    #plt.plot(*drawlines, color='b', lw=2)
    ax.plot3D(*drawlines, color='b', lw = 2)
    plt.show()
