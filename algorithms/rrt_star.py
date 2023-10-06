# File for RRT* algorithm.

# Imports.
import numpy as np
import world

def distance(node_1, node_2):
    return np.linalg.norm(node_1 - node_2, 2, 1)




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
                }
        
    Output arguments:
        nodes - All node positions in tree.
        parents - Parents to each node in tree. 
        costs - Cost to get to each node in tree.
    '''
    
    N = options.get('N')
    npoints = options.get('npoints')
    
    nodes = np.array([snode])    # Specifies all node positions in network.
    parents = np.array([[None]])  # Specifies edges in network.
    costs = np.array([0])    # Specifies costs to get to node.
    
    for i in range(N):
        random_node = sample(world,gnode,options)  # WORLD MEASUREMENTS? ONLY IN FREE SPACE?
        nearest_node, nearest_idx = nearest(nodes, random_node)  # Find nearest node.
        new_node = steer(nearest_node, random_node, options) # Steer towards node and create new node. STEP LENGTH?
        
        if world.obstacle_free(discrete_line(nearest_node, new_node, npoints)): # options?
            new_idx = len(parents) - 1
            neighbor_idxs = neighborhood() # Find neighborhood to new node. RADIUS OF NEIGHBORHOOD?
            min_cost = distance(nearest_node, new_node) + costs[nearest_idx] # Calculate cost to get to new node from nearest. 
            min_idx = nearest_idx
            
            # For all nodes in neighborhood (connect new to cheapest path).
            for neighbor_idx in neighbor_idxs:  
                neighbor_node = nodes[neighbor_idx,:]
                
                if world.obstacle_free(discrete_line(neighbor_node, new_node, npoints)) and ( costs[neighbor_idx] + distance(neighbor_node, new_node) < min_cost ): # IF obstacle free AND cost to get to new node is less than before. 
                    min_cost = costs[neighbor_idx] + distance(neighbor_node, new_node) # Update nearest and cost.
                    min_idx = neighbor_idx
    
            parents.append(min_idx) # Add edge to tree
            
            # For all nodes in neighborhood (rewire tree in neighborhood).
            for neighbor_idx in neighbor_idxs:
                neighbor_node = nodes[neighbor_idx,:]
                
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
        return x


def nearest(nodes, random_node):
    """Find index of state nearest to x in the matrix nodes"""
    nearest_idx = np.argmin(np.sum((nodes - random_node[:, None]) ** 2, axis=0))
    nearest_node = nodes[nearest_idx]
    
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


def neighborhood():

    return


def distance(nearest_node, new_node):
    '''Computing distance between two nodes.'''
    return np.linalg.norm(nearest_node - new_node)


