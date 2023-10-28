import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (0, 0)
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from platoon_gym.envs.utils import HEADWAY_OPTIONS, TOPOLOGY_OPTIONS
from platoon_gym.veh.vehicle import Vehicle
from platoon_gym.veh.virtual_leader import VirtualLeader


class PlatoonEnv(gym.Env):
    """Platoon environment.

    The main platooning gym environment. Visualization is done using pygame.
    Users pass in a list of vehicles and environment arguments. The environment
    arguments contain information about plotting and platoon attributes.

    Attributes:
        vehicles: list[Vehicle], the list of vehicles in the platoon
        virtual_leader: VirtualLeader, the virtual leader
    """

    metadata = {"render_modes": ["plot"], "render_fps": 10}

    def __init__(
        self,
        vehicles: List[Vehicle],
        virtual_leader: VirtualLeader,
        env_args: dict,
        seed: int = 4,
        render_mode: Optional[str] = None,
    ):
        """Initializes the platoon environment.

        Args:
            vehicles: list[Vehicle], the list of vehicles in the platoon
            env_args: dict, environment arguments
            seed: int, random seed
            render_mode: str, the rendering mode
        """
        super().__init__()
        assert env_args["headway"] in HEADWAY_OPTIONS
        assert env_args["topology"] in TOPOLOGY_OPTIONS
        self.env_args = env_args
        self.time = 0.0
        self.dt = env_args["dt"]
        self.n_veh = len(vehicles)
        self.seed = seed
        self.headway = env_args["headway"]

        self.n_plot = env_args["plot history length"]
        self.obs_history = []
        self.err_history = []
        self.state_history = []
        self.time_history = np.array([0.0])

        self.vehicles = vehicles
        self.virtual_leader = virtual_leader

        self.observation_space = spaces.Tuple(
            [
                spaces.Box(
                    low=v.dyn.x_lims[:2, 0],
                    high=v.dyn.x_lims[:2, 1],
                    shape=(2,),
                    dtype=np.float64,
                )
                for v in vehicles
            ]
        )
        self.action_space = spaces.Tuple(
            [
                spaces.Box(
                    low=v.dyn.u_lims[:, 0],
                    high=v.dyn.u_lims[:, 1],
                    shape=(v.dyn.m,),
                    dtype=np.float64,
                )
                for v in vehicles
            ]
        )

        if env_args["headway"] == "constant distance":
            self.d_des = env_args["desired distance"]
            for i in range(len(vehicles)):
                if i == 0:
                    obs = np.array([0.0, 0.0])
                    error = np.array([0.0, 0.0])
                else:
                    distance = (
                        vehicles[i-1].output[0] - vehicles[i].output[0]
                    )
                    position_error = distance - self.d_des
                    velocity_error = vehicles[i-1].output[1] - vehicles[i].output[1]
                    obs = np.array([distance, velocity_error])
                    error = np.array([position_error, velocity_error])
                self.obs_history.append(obs.reshape(-1, 1))
                self.err_history.append(error.reshape(-1, 1))
                self.state_history.append(vehicles[i].state.reshape(-1, 1))
        else:
            raise NotImplementedError

        # rendering stuff
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode:
            self._init_render()

    def _get_obs(self):
        observations = []
        for i in range(len(self.vehicles)):
            if i == 0:
                obs = np.array(
                    [
                        self.virtual_leader.state[0] - self.vehicles[i].output[0],
                        self.virtual_leader.state[1] - self.vehicles[i].output[1]
                    ]
                )
                error = obs
            else:
                distance = (
                    self.vehicles[i-1].output[0] - self.vehicles[i].output[0]
                )
                position_error = distance - self.d_des
                velocity_error = (
                    self.vehicles[i-1].output[1] - self.vehicles[i].output[1]
                )
                obs = np.array([distance, velocity_error])
                error = np.array([position_error, velocity_error])
            observations.append(obs)
            self.obs_history[i] = np.c_[self.obs_history[i], obs]
            self.err_history[i] = np.c_[self.err_history[i], error]
            self.state_history[i] = np.c_[self.state_history[i], self.vehicles[i].state]
            if self.obs_history[i].shape[1] > self.n_plot:
                self.obs_history[i] = self.obs_history[i][:, 1:]
            if self.err_history[i].shape[1] > self.n_plot:
                self.err_history[i] = self.err_history[i][:, 1:]
            if self.state_history[i].shape[1] > self.n_plot:
                self.state_history[i] = self.state_history[i][:, 1:]
        self.time_history = np.concatenate((self.time_history, np.array([self.time])))
        if len(self.time_history) > self.n_plot:
            self.time_history = self.time_history[1:]
        return tuple(observations)

    def _get_info(self):
        return {
            "virtual leader plan": self.virtual_leader.plan,
            "vehicle states": [v.state for v in self.vehicles],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: List[np.ndarray]):
        for i, a in enumerate(action):
            assert a.shape == (self.vehicles[i].m,)
        for i, v in enumerate(self.vehicles):
            v.step(action[i])
        self.virtual_leader.step()
        self.time += self.dt
        obs = self._get_obs()
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "plot":
            self._render_frame()

    def _init_render(self):
        self.plot_size = self.env_args["plot size"]

        self.distance_lines = []
        self.velocity_err_lines = []
        self.position_lines = []
        self.velocity_lines = []

        self.fig, self.ax = plt.subplots(
            nrows=2,
            ncols=2,
            sharex=True,
            figsize=self.plot_size,
            dpi=self.env_args["render dpi"],
        )
        self.fig.subplots_adjust(*self.env_args["subplots adjust"])
        self.fig.suptitle("Platoon dynamics")
        self.ax[0, 0].set_title("spacing error [m]")
        self.ax[0, 1].set_title("velocity difference [m/s]")
        self.ax[1, 0].set_title("position [m]")
        self.ax[1, 1].set_title("velocity [m/s]")
        self.ax[1, 0].set_xlabel("time [s]")
        self.ax[1, 1].set_xlabel("time [s]")
        for i in range(len(self.vehicles)):
            self.distance_lines.append(
                self.ax[0, 0].plot(
                    self.time_history,
                    self.err_history[i][0, :],
                    color=f"C{i}",
                    label=f"{i}",
                )
            )
            self.velocity_err_lines.append(
                self.ax[0, 1].plot(
                    self.time_history, self.err_history[i][1, :], color=f"C{i}"
                )
            )
            self.position_lines.append(
                self.ax[1, 0].plot(
                    self.time_history, self.state_history[i][0, :], color=f"C{i}"
                )
            )
            self.velocity_lines.append(
                self.ax[1, 1].plot(
                    self.time_history, self.state_history[i][1, :], color=f"C{i}"
                )
            )
        self.fig.legend(loc="center right")
        for a in self.ax.flatten():
            a.grid()
        self._set_ax_lims()

        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()

        pygame.init()
        self.window = pygame.display.set_mode(self.raw_data.shape[:2][::-1])
        self.screen = pygame.display.get_surface()
        self.canvas_size = self.canvas.get_width_height()
        self.surf = pygame.image.frombuffer(self.raw_data, self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def _render_frame(self):
        if self.window is None and self.render_mode == "plot":
            pygame.init()
            pygame.display.set_mode(self.raw_data.shape[:2])

        for i in range(len(self.vehicles)):
            self.distance_lines[i] = self.distance_lines[i].pop(0)
            self.distance_lines[i].remove()
            self.distance_lines[i] = self.ax[0, 0].plot(
                self.time_history,
                self.err_history[i][0, :],
                color=f"C{i}",
                label=f"{i}",
            )
            self.velocity_err_lines[i] = self.velocity_err_lines[i].pop(0)
            self.velocity_err_lines[i].remove()
            self.velocity_err_lines[i] = self.ax[0, 1].plot(
                self.time_history, self.err_history[i][1, :], color=f"C{i}"
            )
            self.position_lines[i] = self.position_lines[i].pop(0)
            self.position_lines[i].remove()
            self.position_lines[i] = self.ax[1, 0].plot(
                self.time_history, self.state_history[i][0, :], color=f"C{i}"
            )
            self.velocity_lines[i] = self.velocity_lines[i].pop(0)
            self.velocity_lines[i].remove()
            self.velocity_lines[i] = self.ax[1, 1].plot(
                self.time_history, self.state_history[i][1, :], color=f"C{i}"
            )

        self._set_ax_lims()

        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()
        self.surf = pygame.image.frombuffer(self.raw_data, self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()

        self.clock.tick(self.metadata["render_fps"])

    def _set_ax_lims(self):
        for a in self.ax.flatten():
            a.set_xlim([self.time_history[0], self.time_history[-1] + 1])

        pos_err_lims = (
            min([self.err_history[i][0, :].min() for i in range(len(self.vehicles))])
            - 1,
            max([self.err_history[i][0, :].max() for i in range(len(self.vehicles))])
            + 1,
        )
        vel_err_lims = (
            min([self.err_history[i][1, :].min() for i in range(len(self.vehicles))])
            - 1,
            max([self.err_history[i][1, :].max() for i in range(len(self.vehicles))])
            + 1,
        )
        pos_lims = (
            min([self.state_history[i][0, :].min() for i in range(len(self.vehicles))])
            - 1,
            max([self.state_history[i][0, :].max() for i in range(len(self.vehicles))])
            + 1,
        )
        vel_lims = (
            min([self.state_history[i][1, :].min() for i in range(len(self.vehicles))])
            - 1,
            max([self.state_history[i][1, :].max() for i in range(len(self.vehicles))])
            + 1,
        )
        self.ax[0, 0].set_ylim(pos_err_lims)
        self.ax[0, 1].set_ylim(vel_err_lims)
        self.ax[1, 0].set_ylim(pos_lims)
        self.ax[1, 1].set_ylim(vel_lims)

    def close(self):
        pass
