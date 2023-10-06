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

controller = MPC_controller(A, B, Q, R, N, x0)

# Define the reference trajectory
reference_trajectory = np.array([
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
])

print("Start state: ", controller.x0)

for _ in range(40):
    u = controller.compute_control(reference_trajectory)
    
    print("state: ")
    print(controller.x0)
    print("control: ")
    print(u)
    


