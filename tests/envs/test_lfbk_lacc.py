import gymnasium as gym
import numpy as np
import sys

from platoon_gym.dyn.linear_accel import LinearAccel
from platoon_gym.ctrl.linear_feedback import LinearFeedback
from platoon_gym.veh.vehicle import Vehicle


def test_platoon_env_accel_dyn_lfbk_ctrl():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf],
                       [-np.inf, np.inf],
                       [-np.inf, np.inf]])
    u_lims = np.array([[-5., 5.]])
    dyn = LinearAccel(dt, x_lims, u_lims, tau)

    # set up controller
    if dyn.p == 2:
        k = np.array([[1, 2]])
    elif dyn.p == 3:
        k = np.array([[1, 2, 1]])
    ctrl = LinearFeedback(k)

    # set up platoon env
    if sys.platform.startswith('linux'):
        plot_size = 6
        dpi = 100
        subplots_adjust = [.12, .08, .88, .92, .35, .1]
    elif sys.platform == 'darwin':
        plot_size = 6
        dpi = 50
        subplots_adjust = [.12, .08, .88, .92, .35, .1]
    d_des = 5.0
    env_args = {
        'headway': 'constant distance',
        'desired distance': d_des,
        'topology': 'PF',
        'dt': dt,
        'horizon': None,  # not using MPC method
        'plot history length': 100,
        'init history length': 10,
        'plot size': plot_size,
        'render dpi': dpi,
        'subplots adjust': subplots_adjust
    }
    n_vehicles = 10
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [ctrl for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=20., acceleration=0)]
    vehs += [Vehicle(dyns[i], position=-i*d_des-i, velocity=20., acceleration=0) 
             for i in range(1, n_vehicles)]
    render_mode = 'plot'
    env = gym.make("platoon_gym-v0", vehicles=vehs, env_args=env_args, 
                   render_mode=render_mode)
    obs, info = env.reset()
    action = []
    for o in obs:
        action.append(ctrl.control(o))

    while True:
        obs, reward, terminated, truncated, info = env.step(action=action)
        action = []
        for i, o in enumerate(obs):
            action.append(ctrls[i].control(o))
        env.render()


if __name__ == '__main__':
    test_platoon_env_accel_dyn_lfbk_ctrl()
