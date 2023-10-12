import numpy as np
import cvxpy as cp

class MPC_controller:
    def __init__(self, A, B, Q,  R, N, x0):
        self.A = A
        self.B = B
        self.Q = Q # Weight matrix for state
        self.R = R # Weight matrix for control
        self.N = N # Horizon

        self.x0 = x0.astype(float) # Initial state

        # Define the state and control input dimensions
        self.nr_states = A.shape[0]
        self.nr_controls = B.shape[1]

        # Create optimization variables
        self.X = cp.Variable((self.nr_states, N + 1)) # State, each column is a new time
        self.U = cp.Variable((self.nr_controls, N + 1)) # Control

        self.state_constraints = np.array([[-10, 10],
                                           [-10, 10],
                                           [0, 10]]) # allowed states
        
        self.control_constraints = np.array([[-0.4, 0.4],
                                             [-0.4, 0.4],
                                             [-0.4, 0.4]]) # max and min u


    def compute_control(self, reference_trajectory_N_long):
        # Compute constraints
        constraints = []
        for k in range(self.N):
            if k == 0:
                constraints += [self.X[:, k] == self.x0]

            constraints += [self.X[:, k + 1] == (self.A @ self.X[:, k] + self.B @ self.U[:, k]), self.X[:, k + 1] == (self.A @ self.X[:, k] + self.B @ self.U[:, k])]
            constraints += [self.state_constraints[:, 0] <= self.X[:, k], self.X[:, k] <= self.state_constraints[:, 1]]
            constraints += [self.control_constraints[:, 0] <= self.U[:, k], self.U[:, k] <= self.control_constraints[:, 1]]
        
        self.constraints = constraints

        # Compute cost
        cost = 0.0
        for k in range(self.N):
            cost += cp.quad_form(self.X[:, k] - reference_trajectory_N_long[k, :], self.Q)
            cost += cp.quad_form(self.U[:, k], self.R)
                    
        # Create the optimization problem
        problem = cp.Problem(cp.Minimize(cost), self.constraints)

        # Solve optimatization problem
        problem.solve()

        # Extraxt the first u
        if problem.status == "optimal":
            u = self.U.value[:, 0] # extracts first cloumn
        else:
            u = np.array([0, 0, 0])

        return u

    def set_param(self, A, B, Q,  R, N):
        self.A = A
        self.B = B
        self.Q = Q # Weight matrix for state
        self.R = R # Weight matrix for control
        self.N = N # Horizon

        # Define the state and control input dimensions
        self.nr_states = A.shape[0]
        self.nr_controls = B.shape[1]

        # Create optimization variables
        self.X = cp.Variable((self.nr_states, self.N + 1)) # State
        self.U = cp.Variable((self.nr_controls, self.N)) # Control

    def update_state(self, est_pos):
        self.x0 = est_pos