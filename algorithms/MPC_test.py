import numpy as np
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

    for k in range(len(reference_trajectory) - 10):
        u = controller.compute_control(reference_trajectory[k : k + 10])

        controller.x0 = controller.A @ controller.x0 + controller.B @ u

        print("Error to ", reference_trajectory[k])
        print(controller.x0 - reference_trajectory[k])
