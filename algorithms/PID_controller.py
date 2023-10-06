import numpy as np

class PID_controller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.vmax = 0.4
        self.integral = 0
        self.prev_error = 0
    
    def set_param(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    # Compute velocity (PID controller)
    def compute_control(self, est_pos, pos_des):
        # Errors
        pos_error = est_pos-pos_des
        # Euclidean distance to path
        d = np.linalg.norm(pos_error)
        # Accumulated error
        self.integral += d

        # Compute control law
        v = self.Kp*d + self.Ki*self.integral + self.Kd*(d - self.prev_error)

        # Update previous error
        self.prev_error = d

        # Cap off
        v = min(self.vmax,v)
        vel = v*pos_error/d

        return vel