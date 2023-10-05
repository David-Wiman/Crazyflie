# File for RRT* algorithm.

# Imports.
import numpy as np


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
                }
        
    Output arguments:
        nodes - All node positions in tree.
        parents - Parents to each node in tree. 
        costs - Cost to get to each node in tree.
    '''
    
    N = options.get('N')
    
    nodes = np.array([snode])    # Specifies all node positions in network.
    parents = np.array([[None]])  # Specifies edges in network.
    costs = np.array([0])    # Specifies costs to get to node.
    
    for i in range(N):
        random_node = sample()  # WORLD MEASUREMENTS? ONLY IN FREE SPACE?
        nearest_node, nearest_idx = nearest(nodes, random_node)  # Find nearest node.
        new_node = steer(nearest_node, random_node) # Steer towards node and create new node. STEP LENGTH?
        
        if obstacle_free(world, nearest_node, new_node): # options?
            new_idx = len(parents) - 1
            neighbor_idxs = neighborhood() # Find neighborhood to new node. RADIUS OF NEIGHBORHOOD?
            min_cost = distance(nearest_node, new_node) + costs[nearest_idx] # Calculate cost to get to new node from nearest. 
            min_idx = nearest_idx
            
            # For all nodes in neighborhood (connect new to cheapest path).
            for neighbor_idx in neighbor_idxs:  
                neighbor_node = nodes[neighbor_idx,:]
                
                if obstacle_free(neighbor_node, new_node) and ( costs[neighbor_idx] + distance(neighbor_node, new_node) < min_cost ): # IF obstacle free AND cost to get to new node is less than before. 
                    min_cost = costs[neighbor_idx] + distance(neighbor_node, new_node) # Update nearest and cost.
                    min_idx = neighbor_idx
    
            parents.append(min_idx) # Add edge to tree
            
            # For all nodes in neighborhood (rewire tree in neighborhood).
            for neighbor_idx in neighbor_idxs:
                neighbor_node = nodes[neighbor_idx,:]
                
                # IF collision free AND cost to get to neighborhood node is less via new node:
                if obstacle_free(world, nearest_node, new_node) and cost[new_idx] + distance(neighbor_node, new_node) < cost[neighbor_idx]:
                    parents[neighbor_idx] = new_idx # Set parent for neighborhood node to new node. Remove previous existing edge. Add new edge.
    
            # Checking if new node is the goal node.
            if distance(new_node, gnode) < options.get('terminate_tol'):
                break
            
    # Return tree. 
    return nodes, parents, costs