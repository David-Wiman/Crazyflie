import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import cvxpy as cp
from MPC_controller import MPC_controller

A = np.array([[0.01, 0, 0],
              [0, 0.01, 0],
              [0, 0, 0.01]])

B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Define the cost function (quadratic cost with reference tracking)
Q = np.diag([100.0, 10.0, 2])  # State cost matrix
R = np.diag([0.1, 0.1, 0.1])     # Control cost matrix

N = 10

x0 = np.array([1.5, -0.5, 0.5])

# Define the reference trajectory
reference_trajectory = np.array([
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [1.5, -0.5, 0.5],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
])

if __name__ == '__main__':
    controller = MPC_controller(A, B, Q, R, N, x0)

    print("Start state: ", controller.x0)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for k in range(len(reference_trajectory) - 10):
        u = controller.compute_control(reference_trajectory[k : k + 10])

        #print("Error to ", reference_trajectory[k])
        #print(controller.x0 - reference_trajectory[k])

        ax.plot3D([controller.x0[0]], [controller.x0[1]], [controller.x0[2]], 'o')
        ax.plot3D([reference_trajectory[k][0]], [reference_trajectory[k][1]], [reference_trajectory[k][2]], 'x')
        
    plt.show()