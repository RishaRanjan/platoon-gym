"""Test the platoon environment with linear MPC controller and linear 
velocity dynamics."""

import gymnasium as gym
import numpy as np
import sys
import random
from pathlib import Path

from platoon_gym.dyn.linear_vel import LinearVel
from platoon_gym.ctrl.mpc import LinearMPC
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader
from platoon_gym.veh.utils import VL_TRAJECTORY_TYPES


def generate_data():
    # set up dynamics
    tau = 0.5
    dt = 0.1
    x_lims = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    u_lims = np.array([[-np.inf, np.inf]])
    dyn = LinearVel(dt, x_lims, u_lims, tau)

    # set up controller
    A = dyn.Ad
    B = dyn.Bd
    C = dyn.C
    Q = np.eye(dyn.n)
    R = np.eye(dyn.m)
    Qf = np.eye(dyn.n)
    Cx = np.r_[-np.eye(dyn.n), np.eye(dyn.n)]
    Cu = np.r_[-np.eye(dyn.m), np.eye(dyn.m)]
    dx = np.r_[-dyn.x_lims[:, 0], dyn.x_lims[:, 1]]
    du = np.r_[-dyn.u_lims[:, 0], dyn.u_lims[:, 1]]
    H = 50
    Cf = None
    # Cf = np.eye(dyn.n)
    mpc = LinearMPC(A, B, C, Q, R, Qf, Cx, Cu, dx, du, H, Cf)

    # set up virtual leader
    vl_vel = random.randint(20, 24) 
    vl_traj_type = "random velocity"
    assert vl_traj_type in VL_TRAJECTORY_TYPES
    vl_traj_args = {"horizon": H, "dt": dt}
    vl = VirtualLeader("random velocity", vl_traj_args, velocity=vl_vel)

    # set up platoon env
    subplots_adjust = [0.08, 0.13, 0.88, 0.85, 0.25, 0.3]
    plot_size = (6, 4)
    if sys.platform.startswith("linux"):
        dpi = 100
    elif sys.platform == "darwin":
        dpi = 50
    else:
        exit("Unsupported OS found: {}".format(sys.platform))
    d_des = 5.0
    env_args = {
        "headway": "constant distance",
        "desired distance": d_des,
        "topology": "PF",
        "dt": dt,
        "horizon": H,
        "plot history length": 100,
        "plot size": plot_size,
        "render dpi": dpi,
        "subplots adjust": subplots_adjust,
    }
    n_vehicles = 10
    platoon_vel = 20.0
    dyns = [dyn for _ in range(n_vehicles)]
    ctrls = [mpc for _ in range(n_vehicles)]
    vehs = [Vehicle(dyns[0], position=0, velocity=vl_vel)]
    vehs += [
        Vehicle(dyns[i], position=-i * d_des - i, velocity=platoon_vel)
        for i in range(1, n_vehicles)
    ]
    render_mode = "plot"
    env = gym.make(
        "platoon_gym-v0",
        vehicles=vehs,
        virtual_leader=vl,
        env_args=env_args,
        render_mode=render_mode,
    )
    obs, env_info = env.reset()

    veh_state_plans = []
    veh_control_plans = []
    for v in vehs:
        n, m = v.dyn.n, v.dyn.m
        state_plan = np.zeros((n, H + 1))
        state_plan[:, 0] = v.state[:n]
        control_plan = v.state[1] * np.ones((m, H))
        for k in range(H):
            state_plan[:, k + 1] = v.dyn.forward(state_plan[:, k], control_plan[:, k])
        veh_state_plans.append(state_plan)
        veh_control_plans.append(control_plan)
    veh_states = env_info["vehicle states"]
    vl_plan = env_info["virtual leader plan"][:2, :]
    actions = []
    history = [[] for _ in range(n_vehicles)]
    # number of timesteps recorded in the history
    hist_len = 20
    data = [[] for _ in range(n_vehicles)]
    timestep = 0
    for i, o in enumerate(obs):
        if i == 0:
            action, ctrl_info = ctrls[i].control(
                veh_states[i], vl_plan, veh_control_plans[i]
            )
            vel_err = vl_plan[1][0] - veh_states[i][1]
            pos_err = vl_plan[0][0] - veh_states[i][0]

            veh_err_history = [[pos_err, vel_err] for _ in range(hist_len)]
            veh_err_history = np.array(veh_err_history)
            veh_err_history = veh_err_history.T
            history[i] = veh_err_history
            new_data = np.concatenate(([timestep, veh_states[i][0], veh_states[i][1], action[0]], veh_err_history[0], veh_err_history[1]))
            print(len(new_data))
            assert len(new_data) == 44
            data[i].append(new_data)
        else:
            pred_plan = veh_state_plans[i - 1]
            pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
            action, ctrl_info = ctrls[i].control(
                veh_states[i], pred_plan, veh_control_plans[i]
            )

            env_veh_err_history = env.get_wrapper_attr('err_history')[i]
            pos_err = env_veh_err_history[0][0]
            vel_err = env_veh_err_history[1][0]
            veh_err_history = [[pos_err, vel_err] for _ in range(hist_len)]
            veh_err_history = np.array(veh_err_history)
            veh_err_history = veh_err_history.T
            history[i] = veh_err_history
            new_data = np.concatenate(([timestep, veh_states[i][0], veh_states[i][1], action[0]], veh_err_history[0], veh_err_history[1]))
            assert len(new_data) == 44
            data[i].append(new_data)

            # data[i].append([timestep, veh_states[i][0], veh_states[i][1], action[0]])
        if ctrl_info["status"] != "optimal":
            assert False, f"MPC returned {ctrl_info['status']}"
        veh_state_plans[i] = ctrl_info["planned states"]
        veh_control_plans[i] = ctrl_info["planned inputs"]
        actions.append(action)

# at each time step: numpy array each vehicle's pos and vel and controller input
    while True:
        timestep += 1
        try:
            obs, _, _, _, env_info = env.step(action=actions)
            veh_states = env_info["vehicle states"]
            vl_plan = env_info["virtual leader plan"][:2, :]
            actions = []
            for i, o in enumerate(obs):
                if i == 0:
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i], vl_plan, veh_control_plans[i]
                    )
                    env_veh_err_history = env.get_wrapper_attr('err_history')[i]
                    pos_hist = env_veh_err_history[0][-hist_len:]
                    vel_hist = env_veh_err_history[1][-hist_len:]
                    if (len(pos_hist)< hist_len):
                       pos_err = pos_hist[0] 
                       pos_hist = np.pad(pos_hist, (hist_len-len(pos_hist), 0), 'constant', constant_values=pos_err)
                    if (len(vel_hist)< hist_len):
                       vel_err = vel_hist[0] 
                       vel_hist = np.pad(vel_hist, (hist_len-len(vel_hist), 0), 'constant', constant_values=vel_err)
                       
                    new_data = np.concatenate(([timestep, veh_states[i][0], veh_states[i][1], action[0]], pos_hist, vel_hist))
                    assert len(new_data) == 44
                    data[i].append(new_data)
                else:
                    pred_plan = veh_state_plans[i - 1]
                    pred_plan[0, :] -= pred_plan[0, 0] - veh_states[i][0] - o[0] + d_des
                    action, ctrl_info = ctrls[i].control(
                        veh_states[i], pred_plan, veh_control_plans[i]
                    )
                    env_veh_err_history = env.get_wrapper_attr('err_history')[i]
                    pos_hist = env_veh_err_history[0][-hist_len:]
                    vel_hist = env_veh_err_history[1][-hist_len:]
                    if (len(pos_hist)< hist_len):
                       pos_err = pos_hist[0] 
                       pos_hist = np.pad(pos_hist, (hist_len-len(pos_hist), 0), 'constant', constant_values=pos_err)
                    if (len(vel_hist)< hist_len):
                       vel_err = vel_hist[0] 
                       vel_hist = np.pad(vel_hist, (hist_len-len(vel_hist), 0), 'constant', constant_values=vel_err)
                    new_data = np.concatenate(([timestep, veh_states[i][0], veh_states[i][1], action[0]], pos_hist, vel_hist))
                    assert len(new_data) == 44
                    data[i].append(new_data)
                if ctrl_info["status"] != "optimal":
                    assert False, f"MPC returned {ctrl_info['status']}"
                veh_state_plans[i] = ctrl_info["planned states"]
                veh_control_plans[i] = ctrl_info["planned inputs"]
                actions.append(action)
            env.render()
        except KeyboardInterrupt:
            env.close()
            for i in range(n_vehicles):
                veh_arr = np.array(data[i])
                data_dir = get_project_root() + '/data' + f'/veh_{i}.npy'
                data_dir2 = get_project_root() + '/data' + f'/veh_{i}.csv'
                np.save(data_dir, veh_arr)
                np.savetxt(data_dir2, veh_arr)
            break


def get_project_root() -> str:
    """
    Returns:
        str: project root path
    """
    return str(Path(__file__).parent.parent)

if __name__ == "__main__":
    generate_data()


