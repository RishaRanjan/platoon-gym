import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from platoon_gym.envs.utils import HEADWAY_OPTIONS, TOPOLOGY_OPTIONS
from platoon_gym.veh.vehicle import Vehicle


class PlatoonEnv(gym.Env):
    metadata = {'render_modes': ['plot'], 'render_fps': 10}

    def __init__(self, 
                 vehicles: List[Vehicle],
                 env_args: dict,
                 seed: int = 4, 
                 render_mode: Optional[str] = None):
        super().__init__()
        assert env_args['headway'] in HEADWAY_OPTIONS
        assert env_args['topology'] in TOPOLOGY_OPTIONS
        self.env_args = env_args
        self.time = 0.
        self.dt = env_args['dt']
        self.n_veh = len(vehicles)
        self.seed = seed
        self.headway = env_args['headway']

        self.n_plot = env_args['plot history length']
        self.err_history = []
        self.state_history = []
        self.time_history = np.array([0.])
        self.position_err_lines = [None for v in vehicles]
        self.velocity_err_lines = [None for v in vehicles]
        self.position_lines = [None for v in vehicles]
        self.velocity_lines = [None for v in vehicles]

        self.vehicles = vehicles

        self.observation_space = spaces.Tuple(
            [spaces.Box(low=v.dyn.x_lims[:v.dyn.p, 0], 
                        high=v.dyn.x_lims[:v.dyn.p, 1],
                        shape=(v.dyn.p,), dtype=np.float64) for v in vehicles])
        self.action_space = spaces.Tuple(
            [spaces.Box(low=v.dyn.u_lims[:, 0],
                        high=v.dyn.u_lims[:, 1],
                        shape=(v.dyn.m,), dtype=np.float64) for v in vehicles])

        # TODO: add leader reference trajectory
        if env_args['headway'] == 'constant distance':
            self.d_des = env_args['desired distance']
            for i in range(len(vehicles)):
                if i == 0:
                    error = np.array([0., 0.])
                else:
                    position_error = vehicles[i].output[0] - \
                        vehicles[i-1].output[0] + self.d_des
                    velocity_error = vehicles[i].output[1] - \
                        vehicles[i-1].output[1]
                    error = np.array([position_error, velocity_error])
                self.err_history.append(error.reshape(-1, 1))
                self.state_history.append(vehicles[i].state.reshape(-1, 1))

        # rendering stuff
        assert render_mode is None or \
               render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        if self.render_mode:
            self._init_render()
        
    def _get_obs(self):
        # TODO: add leader reference trajectory
        self.errors = []
        for i in range(len(self.vehicles)):
            if i == 0: 
                error = np.array([0., 0.])
            else:
                position_error = self.vehicles[i].output[0] - \
                                self.vehicles[i-1].output[0] + self.d_des
                velocity_error = self.vehicles[i].output[1] - \
                                 self.vehicles[i-1].output[1]
                error = np.array([position_error, velocity_error])
            self.errors.append(error)
            self.err_history[i] = np.c_[self.err_history[i], error]
            self.state_history[i] = np.c_[self.state_history[i], 
                                          self.vehicles[i].state]
            if self.err_history[i].shape[1] > self.n_plot:
                self.err_history[i] = self.err_history[i][:, 1:]
            if self.state_history[i].shape[1] > self.n_plot:
                self.state_history[i] = self.state_history[i][:, 1:]
        self.time_history = \
            np.concatenate((self.time_history, np.array([self.time])))
        if len(self.time_history) > self.n_plot:
            self.time_history = self.time_history[1:]
        return tuple(self.errors)

    def _get_info(self):
        return {}

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
        self.time += self.dt
        obs = self._get_obs()
        reward = 0.
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'plot':
            self._render_frame()
    
    def _init_render(self):
        self.plot_size = self.env_args['plot size']

        self.fig, self.ax = plt.subplots(
            nrows=2, ncols=2, sharex='col', 
            figsize=(self.plot_size, self.plot_size),
            dpi=self.env_args['render dpi']
        )
        self.fig.subplots_adjust(*self.env_args['subplots adjust'])
        self.fig.suptitle("Platoon dynamics")
        self.ax[0, 0].set_ylabel("position error [m]")
        self.ax[0, 1].set_ylabel("velocity error [m/s]")
        self.ax[1, 0].set_ylabel("position [m]")
        self.ax[1, 1].set_ylabel("velocity [m/s]")
        self.ax[1, 0].set_xlabel("time [s]")
        self.ax[1, 1].set_xlabel("time [s]")
        for i in range(len(self.vehicles)):
            self.position_err_lines[i] = \
                self.ax[0, 0].plot(self.time_history, self.err_history[i][0, :], 
                                color=f"C{i}", label=f"{i}")
            self.velocity_err_lines[i] = \
                self.ax[0, 1].plot(self.time_history, self.err_history[i][1, :], 
                                color=f"C{i}", label=f"{i}")
            self.position_lines[i] = \
                self.ax[1, 0].plot(self.time_history, self.state_history[i][0, :], 
                                color=f"C{i}", label=f"{i}")
            self.velocity_lines[i] = \
                self.ax[1, 1].plot(self.time_history, self.state_history[i][1, :],
                                   color=f"C{i}", label=f"{i}")
        self.ax[0, 1].legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
        self.ax[1, 1].legend(bbox_to_anchor=(1.0, 0.5), loc='center left')
        for a in self.ax.flatten():
            a.grid()
            a.set_xlim([self.time_history[0], self.time_history[-1] + 1])

        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()

        pygame.init()
        self.window = pygame.display.set_mode(self.raw_data.shape[:2])
        self.screen = pygame.display.get_surface()
        self.canvas_size = self.canvas.get_width_height()
        self.surf = pygame.image.frombuffer(self.raw_data, 
                                            self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def _render_frame(self):
        if self.window is None and self.render_mode == 'plot':
            pygame.init()
            pygame.display.set_mode(self.raw_data.shape[:2])

        # TODO: update lines in plot
        for i in range(len(self.vehicles)):
            self.position_err_lines[i] = self.position_err_lines[i].pop(0)
            self.position_err_lines[i].remove()
            self.position_err_lines[i] = \
                self.ax[0, 0].plot(self.time_history, self.err_history[i][0, :], 
                                   color=f"C{i}", label=f"{i}")
            self.velocity_err_lines[i] = self.velocity_err_lines[i].pop(0)
            self.velocity_err_lines[i].remove()
            self.velocity_err_lines[i] = \
                self.ax[0, 1].plot(self.time_history, self.err_history[i][1, :], 
                                   color=f"C{i}", label=f"{i}")
            self.position_lines[i] = self.position_lines[i].pop(0)
            self.position_lines[i].remove()
            self.position_lines[i] = \
                self.ax[1, 0].plot(self.time_history, 
                                   self.state_history[i][0, :], color=f"C{i}",
                                   label=f"{i}")
            self.velocity_lines[i] = self.velocity_lines[i].pop(0)
            self.velocity_lines[i].remove()
            self.velocity_lines[i] = \
                self.ax[1, 1].plot(self.time_history,
                                   self.state_history[i][1, :], color=f"C{i}",
                                   label=f"{i}")
        
        for a in self.ax.flatten():
            a.set_xlim([self.time_history[0], self.time_history[-1] + 1])

        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()
        self.raw_data = self.renderer.buffer_rgba()
        self.surf = pygame.image.frombuffer(self.raw_data, 
                                            self.canvas_size, "RGBA")
        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()

        self.clock.tick(self.metadata['render_fps'])

    def close(self):
        pass