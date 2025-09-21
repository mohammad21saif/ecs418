import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import math

import warnings

warnings.filterwarnings('ignore')


class ConfigurationSpace:
    def __init__(self, start_pos, goal_pos, init_pos):
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.init_pos = init_pos

        self.x_min, self.x_max = 0, 51
        self.y_min, self.y_max = 0, 51

    def visualize(self, path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')

        ax.plot(
            [self.start_pos[0], self.goal_pos[0]],
            [self.start_pos[1], self.goal_pos[1]],
            color='black',
            linewidth=3,
            label='Start to Goal'
        )

        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=12, label='Initial Position')
        ax.plot(self.goal_pos[0], self.goal_pos[1], 'ro', markersize=12, label='Goal Position')

        if path is not None:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vector Field Path Following')
        ax.legend()
        plt.tight_layout()
        plt.savefig('vector_field_path.png')
        plt.show()

class VectorField:
    def __init__(self, config_space, heading, entry_heading, tau, k, kappa, alpha, v=1, dt=0.1):
        self.config_space = config_space
        self.heading = heading
        self.entry_heading = entry_heading
        self.tau = tau
        self.k = k
        self.kappa = kappa
        self.alpha = alpha
        self.v = v
        self.dt = dt
        self.current_pos = config_space.init_pos
        self.path = [self.current_pos]

    def vector_field(self, max_iters=1000):
        for i in range(max_iters):
            x, y = self.current_pos
            gx, gy = self.config_space.goal_pos

            dist_to_goal = math.hypot(gx - x, gy - y)
            if dist_to_goal < 0.5:
                print("Reached the goal!")
                break

            theta = math.atan2(gy - self.config_space.start_pos[1], gx - self.config_space.start_pos[0])
            
