""" David Wiman, 17/10 2023, as part of TSFS12 at LiU """

# Imports
import numpy as np
import cvxpy as cp

class MPC_controller:
    def __init__(self, A, B, Q,  R, N, state, sampling_time):
        self.A = A # State transition matrix
        self.B = B*sampling_time
        self.Q = Q # Weight matrix for state
        self.R = R # Weight matrix for control
        self.N = N # Horizon
        self.u = 0 # Last control signal

        self.state = state.astype(float) # Initial state

        # Define the state and control input dimensions
        self.nr_states = A.shape[0]
        self.nr_controls = B.shape[1]

        # Create optimization variables
        self.X = cp.Variable((self.nr_states, N + 1)) # State, each column is a new time
        self.U = cp.Variable((self.nr_controls, N + 1)) # Control

        # Create constraints, change for you setup
        self.state_constraints = np.array([[-2, 2],
                                           [-2, 2],
                                           [0, 2]]) # allowed states
        
        self.control_constraints = np.array([[-0.4, 0.4],
                                             [-0.4, 0.4],
                                             [-0.4, 0.4]]) # max and min u

    """Main method, returns control signal u given a 'horizon long' trajectory"""
    def compute_control(self, reference_trajectory_N_long):
        """Compute constraints"""
        constraints = []
        tolerance = np.array([1, 1, 1])*0.01
        for k in range(self.N):
            if k == 0:
                # First state must be current state
                constraints += [self.X[:, k] == self.state]
            # Each consecutive state must be a natural continuation of the previous
            constraints += [self.X[:, k + 1] - (self.A @ self.X[:, k] + self.B @ self.U[:, k]) <= tolerance, -tolerance <= self.X[:, k + 1] - (self.A @ self.X[:, k] + self.B @ self.U[:, k])]
            # Only fly in the allowed box
            constraints += [self.state_constraints[:, 0] <= self.X[:, k], self.X[:, k] <= self.state_constraints[:, 1]]
            # Only use the allowed control signals
            constraints += [self.control_constraints[:, 0] <= self.U[:, k], self.U[:, k] <= self.control_constraints[:, 1]]
        
        self.constraints = constraints

        """Compute cost"""
        cost = 0.0
        for k in range(self.N):
            # Deviating from reference
            cost += cp.quad_form(self.X[:, k] - reference_trajectory_N_long[k, :], self.Q)
            # Using engines
            cost += cp.quad_form(self.U[:, k], self.R)
                    
        # Create the optimization problem
        problem = cp.Problem(cp.Minimize(cost), self.constraints)

        # Solve optimatization problem
        problem.solve()

        # Extraxt the first u
        if problem.status == "optimal":
            u = self.U.value[:, 0] # extracts first cloumn
        else:
            u = np.array([0, 0, 0]) # deafult if optimazation failed

        # Update last known u
        self.u = u

        return u

    """Method for using external measurement data to update the state"""
    def update_state(self, est_pos):
        self.state = est_pos

    """Method for internaly updating the state"""
    def internal_update_state(self, est_pos):
        self.state = self.A @ self.state + self.B @ self.u