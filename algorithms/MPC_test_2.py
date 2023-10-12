import numpy as np
import cvxpy as cp
from MPC_controller import MPC_controller
import time

A = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

B = 10*np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])

# Define the cost function (quadratic cost with reference tracking)
Q = np.diag([100, 100, 100])  # State cost matrix
R = np.diag([0.1, 0.1, 0.1])     # Control cost matrix

N = 10

controller = MPC_controller(A, B, Q, R, N, x0=np.array([1.5, -0.5, 0.5]))

startTime = time.time()
ref_index = 0
sequence = np.array([
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

position = sequence[ref_index]

for _ in range(100):
    ref_index += 1

    if ref_index >= len(sequence):
        break

    position = sequence[ref_index]

    reference_trajectory = sequence

    # At the end of the sequence, add multiples of the last element
    if (ref_index + controller.N >= sequence.shape[0]):
        added_points = np.repeat([sequence[-1]], controller.N, axis=0)

        reference_trajectory = np.vstack([sequence, added_points])

    print("Tracking point number ", ref_index, " out of ", len(sequence))

    u = controller.compute_control(reference_trajectory[ref_index:ref_index + controller.N])

    est_pos = controller.A @ controller.x0 + controller.B @ u

    controller.update_state(est_pos)