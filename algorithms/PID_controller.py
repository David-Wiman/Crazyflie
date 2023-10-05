class PID_controller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.vmax = 0.4
        self.integral = 0
        self.prev_error = 0
    
    def set_param(self, Kp, Ki, Kd)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    # Compute velocity (PID controller)
    def compute_control(self, x, y, z, x_des, y_des, z_des)
        # Errors
        xe = x-x_des
        ye = y-y_des
        ze = z-z_des

        # Euclidean distance to path
        d = math.sqrt(xe**2+ye**2+ze**2)

        # Accumulated error
        self.integral += d

        # Compute control law
        v = self.Kp*d + self.Ki*self.integral + self.Kd*(d - self.prev_error)

        # Update previous error
        self.prev_error = d

        # Cap off
        v = min(self.vmax, v)

        # Normalize
        vx = -v*xe/d
        vy = -v*ye/d
        vz = -v*ze/d

        return vx, vy, 