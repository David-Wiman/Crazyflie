# -*- coding: utf-8 -*-
#
"""
Simple example that connects to one crazyflie, sets the initial position
flies towards specified positions in sequence using onboard velocity control.
Using synchronous crazyflie classes suitable to control a single drone.
Works best with lighthouse/loco positioning systems.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

# Primarily see __init__.py in cfsim/crazyflie/ to add functionality to simulator
from algorithms.PID_controller import PID_controller
from algorithms.world import BoxWorld, create_mission
from algorithms.rrt_star import rrt_star
from algorithms.MPC_controller import MPC_controller
from mpl_toolkits.mplot3d import Axes3D

simulate = True

if simulate:
    import cfsim.crtp as crtp
    from cfsim.crazyflie import Crazyflie
    from cfsim.crazyflie.log import LogConfig
    from cfsim.crazyflie.syncCrazyflie import SyncCrazyflie
    from cfsim.crazyflie.syncLogger import SyncLogger
else:
    import cflib.crtp as crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.syncLogger import SyncLogger


def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]
            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break


def set_initial_position(scf):
    scf.cf.param.set_value('kalman.initialX', start_node[0])
    scf.cf.param.set_value('kalman.initialY', start_node[1])
    scf.cf.param.set_value('kalman.initialZ', start_node[2])

def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)


def run_sequence(scf, logger, sequence, goal_node, tolerance, sampling_time):
    cf = scf.cf

    startTime = time.time()
    sequencePos = 0
    previous_time = 0

    # Set starting position
    position = sequence[sequencePos]
    print('Setting reference position {}'.format(position))

    for log_entry in logger: #Synchronous list (runs when a new position is recieved, otherwise blocks)
        timestamp = log_entry[0]
        data = log_entry[1]
        logconf_name = log_entry[2]

        #print('[%d][%s]: %s' % (timestamp, logconf_name, data))

        # Determine position reference based on time
        relativeTime = time.time()-startTime
        if relativeTime > (previous_time+sampling_time):# Fly to each point for 5 seconds
            sequencePos = np.argmin(np.sum((sequence - controller.x0) ** 2, axis=1))+1
            previous_time = relativeTime

            if sequencePos >= len(sequence):
                break

            position = sequence[sequencePos]
            print('Setting reference position {}'.format(position))

        reference_trajectory = sequence
        
        # At the end of the sequence, add multiples of the last element
        if (sequencePos + controller.N >= sequence.shape[0]):
            added_points = np.repeat([sequence[-1]], controller.N, axis=0)

            reference_trajectory = np.vstack([sequence, added_points])

        vel = controller.compute_control(reference_trajectory[sequencePos:sequencePos + controller.N])

        cf.commander.send_velocity_world_setpoint(vel[0], vel[1], vel[2], 0)

        est_pos = np.array([data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ']])
        print(f"Estimated position: {est_pos}, dist: {np.linalg.norm(est_pos - goal_node)}")
        if np.linalg.norm(est_pos - goal_node) < tolerance or relativeTime > 60:
            break

        controller.update_state(est_pos)

        # Log some data
        logdata[uri]['x'].append(controller.x0[0])
        logdata[uri]['y'].append(controller.x0[1])
        logdata[uri]['z'].append(controller.x0[2])
        

    print('Landing')
    for i in range(20):
        cf.commander.send_velocity_world_setpoint(0, 0, -0.1, 0)
        time.sleep(0.1)
    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)

def plot_path(logdata, path, world):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(projection='3d')
    plt.xlim([world.xmin, world.xmax])
    plt.ylim([world.ymin, world.ymax])
    #plt.zlim([world.zmin, world.zmax])
    plt.plot(logdata[uri]['x'], logdata[uri]['y'], logdata[uri]['z'], 'b')
    plt.plot(*path.T, '--r')
    plt.legend(["Real Position", "Planned Path"])
    plt.show()


if __name__ == '__main__':
    logdata = {}
    crazy_flie_nbr = 1

    # URI to the Crazyflie to connect to
    uri = f'radio://0/80/2M/E7E7E7E70{crazy_flie_nbr}'

    logdata[uri] = {'x':[],'y':[],'z':[]}

    world = BoxWorld([[-2, 2], [-1, 1], [0, 2]])
    world = create_mission(world, 1)
    start_node = np.array([0, 0, 0])
    goal_node = np.array([0, 0, 1.7])
    options = {
            'N': 10000,
            'terminate_tol': 0.01,
            'npoints': 50,
            'beta': 0.15,
            'lambda': 0.01,
            'r': np.sqrt(0.01),
            'full_search' : False,
    }

    # path, nodes, parents, costs, goal_node_index = rrt_star(start_node, goal_node, world, options)
    path = np.array([
        [-5.69767701e-02,  1.00512585e-03,  4.86117567e-02],
        [-9.51935405e-02,  4.63777938e-03,  7.50264374e-02],
        [-1.57517548e-01,  1.19982652e-02,  1.15520835e-01],
        [-2.30863440e-01, -1.99338298e-03,  1.75048541e-01],
        [-2.94523714e-01, -1.53145613e-02,  2.18340635e-01],
        [-3.37105760e-01, -2.44143129e-02,  2.32241617e-01],
        [-4.09058292e-01, -3.27786826e-02,  2.41833969e-01],
        [-4.97346872e-01, -4.89312467e-02,  2.47913182e-01],
        [-5.29337430e-01, -5.68141553e-02,  2.49271221e-01],
        [-6.19516060e-01, -4.43180499e-02,  2.45976958e-01],
        [-6.42838556e-01, -3.71077668e-02,  2.49408358e-01],
        [-7.30898178e-01,  1.68660655e-03,  2.42521128e-01],
        [-7.80113887e-01,  3.20361129e-02,  2.49491107e-01],
        [-8.77793949e-01,  3.44137685e-02,  2.45068981e-01],
        [-9.12634920e-01,  2.22643572e-02,  2.42031635e-01],
        [-1.00516266e+00,  1.33330693e-03,  2.48671995e-01],
        [-1.04533087e+00,  3.76767328e-03,  3.16666320e-01],
        [-1.06109770e+00,  9.61206811e-03,  3.48844567e-01],
        [-1.09757940e+00, -9.28393909e-04,  4.38893408e-01],
        [-1.13003111e+00, -1.91688148e-02,  5.30076693e-01],
        [-1.13083450e+00, -2.01383345e-02,  5.63044845e-01],
        [-1.14219349e+00, -2.38205874e-02,  6.56448864e-01],
        [-1.17008285e+00, -4.02550513e-02,  7.49183333e-01],
        [-1.12071579e+00, -4.88594514e-02,  8.26905893e-01],
        [-1.07378852e+00, -5.06948385e-02,  8.72461705e-01],
        [-1.03302535e+00, -4.51698601e-02,  9.18022617e-01],
        [-9.71332048e-01, -4.80300514e-02,  9.79245976e-01],
        [-9.10742522e-01, -4.19695193e-02,  1.04596465e+00],
        [-8.30714460e-01, -3.15043735e-02,  1.09806391e+00],
        [-8.22620634e-01, -3.11974198e-02,  1.10392870e+00],
        [-7.38300191e-01, -2.51202567e-02,  1.15258370e+00],
        [-6.80336561e-01, -1.80433779e-02,  1.20514942e+00],
        [-6.23484162e-01, -2.68280405e-04,  1.24838381e+00],
        [-5.67044786e-01,  4.47059285e-03,  1.27991322e+00],
        [-5.08720010e-01, -1.66007020e-02,  1.34949535e+00],
        [-5.01442838e-01, -2.86362716e-02,  1.36581412e+00],
        [-5.05651552e-01, -7.99460505e-02,  1.44969302e+00],
        [-5.11529857e-01, -1.19400021e-01,  1.48955579e+00],
        [-4.95768772e-01, -1.25066233e-01,  1.50441278e+00],
        [-4.44606607e-01, -1.13491651e-01,  1.53020406e+00],
        [-3.61674758e-01, -1.03759722e-01,  1.57410782e+00],
        [-3.52559198e-01, -1.01144588e-01,  1.57728077e+00],
        [-3.43443638e-01, -9.85294537e-02,  1.58045373e+00],
        [-3.34328078e-01, -9.59143197e-02,  1.58362668e+00],
        [-3.25212518e-01, -9.32991857e-02,  1.58679964e+00],
        [-3.16096959e-01, -9.06840517e-02,  1.58997259e+00],
        [-3.06981399e-01, -8.80689177e-02,  1.59314555e+00],
        [-2.97865839e-01, -8.54537838e-02,  1.59631850e+00],
        [-2.88750279e-01, -8.28386498e-02,  1.59949146e+00],
        [-2.79634719e-01, -8.02235158e-02,  1.60266441e+00],
        [-1.96745105e-01, -6.27191884e-02,  1.63480355e+00],
        [-1.33146384e-01, -4.24449348e-02,  1.65587859e+00],
        [-1.24060853e-01, -3.95486128e-02,  1.65888931e+00],
        [-1.14975321e-01, -3.66522909e-02,  1.66190003e+00],
        [-1.05889790e-01, -3.37559689e-02,  1.66491075e+00],
        [-9.68042581e-02, -3.08596470e-02,  1.66792147e+00],
        [-8.77187265e-02, -2.79633250e-02,  1.67093219e+00]])

    new_path = path[0]
    for i in range(len(path)-1):
        new_path = np.vstack([new_path, np.linspace(path[i], path[i+1], 10, axis = 0).squeeze()[1:]])

    path = new_path
    
    # Evaluate and plot.
   #print(f'Number of nodes {len(parents)} or {nodes.shape}, number of nodes on path {path.shape}')

    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    B = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

    # Define the cost function (quadratic cost with reference tracking)
    Q = np.diag([10, 10, 10])  # State cost matrix
    R = np.diag([0.1, 0.1, 0.1])     # Control cost matrix

    N = 5
    sampling_time=0.1

    controller = MPC_controller(A, B, Q, R, N, start_node, sampling_time)

    crtp.init_drivers(enable_debug_driver=False)

    # Set logging ()
    log_config = LogConfig(name='Position', period_in_ms=50)
    log_config.add_variable('kalman.stateX', 'float')
    log_config.add_variable('kalman.stateY', 'float')
    log_config.add_variable('kalman.stateZ', 'float')

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        set_initial_position(scf)
        reset_estimator(scf)

        with SyncLogger(scf, log_config) as logger:
            run_sequence(scf, logger, path, goal_node, options["terminate_tol"], sampling_time)
    plot_path(logdata, path, world)