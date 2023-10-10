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
from algorithms.world import BoxWorld
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


def run_sequence(scf, logger, sequence):
    cf = scf.cf

    startTime = time.time()
    sequencePos = 0

    # Set starting position
    position = sequence[sequencePos]
    print('Setting position {}'.format(position))

    for log_entry in logger: #Synchronous list (runs when a new position is recieved, otherwise blocks)
        timestamp = log_entry[0]
        data = log_entry[1]
        logconf_name = log_entry[2]

        #print('[%d][%s]: %s' % (timestamp, logconf_name, data))

        # Determine position reference based on time
        relativeTime = time.time()-startTime
        if relativeTime > (sequencePos+1)*0.1: # Fly to each point for 5 seconds
            sequencePos += 1

            if sequencePos >= len(sequence):
                break

            position = sequence[sequencePos]
            print('Setting position {}'.format(position))

        reference_trajectory = np.repeat(position[np.newaxis,...], 20, axis=0)

        vel = controller.compute_control(reference_trajectory)

        x0_list.append(controller.x0)

        cf.commander.send_velocity_world_setpoint(vel[0], vel[1], vel[2], 0)

        # Log some data
        logdata[uri]['x'].append(position[0])
        logdata[uri]['y'].append(position[1])
        logdata[uri]['z'].append(position[2])

    print('Landing')
    for i in range(20):
        cf.commander.send_velocity_world_setpoint(0, 0, -0.1, 0)
        time.sleep(0.1)
    cf.commander.send_stop_setpoint()
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)

def plot_path(logdata):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.add_subplot(projection='3d')
    plt.plot(logdata[uri]['x'], logdata[uri]['y'], logdata[uri]['z'], 'b')


if __name__ == '__main__':
    logdata = {}
    crazy_flie_nbr = 1

    world = BoxWorld()
    start_node = np.array([0,0,0])

    start_node = np.array([0, 0, 0])
    goal_node = np.array([10, 5, 2])
    options = {
            'N': 10000,
            'terminate_tol': 0.1,
            'npoints': 50,
            'beta': 0.05,
            'lambda': 0.3,
            'r': np.sqrt(0.4),
    }

    # URI to the Crazyflie to connect to
    uri = f'radio://0/80/2M/E7E7E7E70{crazy_flie_nbr}'

    logdata[uri] = {'x':[],'y':[],'z':[]}

    world = BoxWorld()

    start_node = np.array([0, 0, 0])
    goal_node = np.array([10, 10, 5])
    options = {
        'N': 10000,
        'terminate_tol': 0.1,
        'npoints': 50,
        'beta': 0.05,
        'lambda': 0.3,
        'r': np.sqrt(0.4),
    }

    path, sequence, parents, costs = rrt_star(start_node, goal_node, world, options)

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

    controller = MPC_controller(A, B, Q, R, N, x0=np.array([0, 0, 0]))

    x0_list = []

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
            run_sequence(scf, logger, path)

    plot_path(logdata)
    plt.plot(*path.T, '--r')
    plt.legend(["Real Position", "Planned Path"])
    plt.show()