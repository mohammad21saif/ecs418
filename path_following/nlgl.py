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
        ax.set_title('NLGL Path Following')
        ax.legend()
        plt.tight_layout()
        plt.savefig('nlgl_path.png')
        plt.show()


class NLGL:
    def __init__(self, config_space, heading, L=20, v=1, dt=0.1):
        self.config_space = config_space
        self.heading = heading  
        self.L = L              
        self.v = v              
        self.dt = dt
        self.current_pos = config_space.init_pos
        self.path = [self.current_pos]

    def give_target_point(self, x, y):
        dx = self.config_space.goal_pos[0] - self.config_space.start_pos[0]
        dy = self.config_space.goal_pos[1] - self.config_space.start_pos[1]

        A = dx**2 + dy**2
        B = 2 * (dx * (self.config_space.start_pos[0] - x) +
                 dy * (self.config_space.start_pos[1] - y))
        C = ((self.config_space.start_pos[0] - x)**2 +
             (self.config_space.start_pos[1] - y)**2 - self.L**2)

        discriminant = B**2 - 4 * A * C
        if discriminant < 0:
            return None

        t1 = (-B + math.sqrt(discriminant)) / (2 * A)
        t2 = (-B - math.sqrt(discriminant)) / (2 * A)

        candidates = []
        for t in [t1, t2]:
            if 0 <= t <= 1:
                tx = self.config_space.start_pos[0] + t * dx
                ty = self.config_space.start_pos[1] + t * dy
                candidates.append((tx, ty))

        if not candidates:
            return None

        return max(candidates, key=lambda p: p[0])


    def nlgl(self, max_iters=1000):
        for i in range(max_iters):
            x, y = self.current_pos
            gx, gy = self.config_space.goal_pos

            dist_to_goal = math.hypot(gx - x, gy - y)
            if dist_to_goal < 0.5:
                print("Reached the goal!")
                break

            target = self.give_target_point(x, y)
            if target is None:
                print("No valid target point")
                break

            theta = math.atan2(target[1] - y, target[0] - x)
            eta = theta - self.heading
            eta = (eta + math.pi) % (2 * math.pi) - math.pi 

            u = (2 * self.v / self.L) * math.sin(eta) 

            self.heading += u * self.dt
            x += self.v * math.cos(self.heading) * self.dt
            y += self.v * math.sin(self.heading) * self.dt

            self.current_pos = (x, y)
            self.path.append(self.current_pos)

        return self.path

def main():
    start_pos = (5, 5)
    goal_pos = (45, 45)
    init_pos = (10, 8)
    config_space = ConfigurationSpace(start_pos, goal_pos, init_pos)

    L = 10.0
    v = 1.0
    heading = math.radians(90)  

    nlgl = NLGL(config_space, heading, L, v)
    path = nlgl.nlgl()

    config_space.visualize(path)

if __name__ == '__main__':
    main()


