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

        # Path
        if path is not None:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')

        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Carrot Chasing Path Following')
        ax.legend()
        plt.tight_layout()
        plt.savefig('carrot_chase_path.png')
        plt.show()




class CarrotChase:
    def __init__(self, config_space, delta, k, heading, v=1.0, dt=0.1):
        self.config_space = config_space
        self.current_pos = config_space.init_pos
        self.path = [self.current_pos]
        self.delta = delta
        self.k = k
        self.heading = heading # in degrees
        self.v = v
        self.dt = dt

    def carrot_chase(self, max_iters=1000):
        for i in range(max_iters):
            x, y = self.current_pos
            gx, gy = self.config_space.goal_pos

            dist_to_goal = math.hypot(gx - x, gy - y)
            if dist_to_goal < 0.5: 
                print("Reached the goal!")
                break

            theta = math.atan2(gy - self.config_space.start_pos[1],
                               gx - self.config_space.start_pos[0])

            Ru = math.hypot(x - self.config_space.start_pos[0],
                            y - self.config_space.start_pos[1])
            thetau = math.atan2(y - self.config_space.start_pos[1],
                                x - self.config_space.start_pos[0])

            beta = theta - thetau
            R = math.sqrt(Ru**2 + (Ru * math.sin(beta))**2)
            x_i = (R + self.delta) * math.cos(theta) + self.config_space.start_pos[0]
            y_i = (R + self.delta) * math.sin(theta) + self.config_space.start_pos[1]

            desired_heading = math.atan2(y_i - y, x_i - x)
            heading_error = desired_heading - self.heading

            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

            omega = self.k * heading_error

            self.heading += omega * self.dt
            x = x + self.v * math.cos(self.heading) * self.dt
            y = y + self.v * math.sin(self.heading) * self.dt
            self.current_pos = (x, y)
            self.path.append(self.current_pos)
        
        return self.path
    


def main():
    start_pos = (5, 5)
    goal_pos = (45, 45)
    init_pos = (30, 10)
    config_space = ConfigurationSpace(start_pos, goal_pos, init_pos)

    delta = 2.0
    k = 1.0
    heading = math.radians(-90)  

    carrot_chase = CarrotChase(config_space, delta, k, heading)
    path = carrot_chase.carrot_chase()

    config_space.visualize(path)


if __name__ == "__main__":
    main()


            


        