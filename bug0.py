import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import math
from enum import Enum
import warnings
import time  

warnings.filterwarnings('ignore')


class ConfigurationSpace:
    def __init__(self):
        self.x_min, self.x_max = 0, 31
        self.y_min, self.y_max = 0, 31

        self.start = (0, 0)
        self.goal = (20, 20)

        self.obstacles = [
            (4.5, 3, 2),
            (3, 12, 2),
            (15, 15, 3)
        ]

    def check_collision(self, x, y, margin=0.1) -> bool:
        '''
        Check if a point (x, y) collides with any obstacles in the configuration space.
        '''
        for obs_x, obs_y, radius in self.obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance <= radius + margin:
                return True
        return False

    def check_valid_point(self, x, y) -> bool:
        '''
        Check if a point (x, y) is within bounds and not in collision.
        '''
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        return not self.check_collision(x, y)


    def distance_to_goal(self, x, y) -> float:
        '''
        Calculate the Euclidean distance from a point (x, y) to the goal.
        '''
        return math.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)


    def visualize(self, path=None) -> None:
        '''
        Visualize the configuration space, obstacles, start and goal points, and the path taken by Bug0.
        '''
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')

        for i, (obsx, obsy, r) in enumerate(self.obstacles):
            circle = Circle((obsx, obsy), r, color='red', alpha=0.7, label=f"Obstacle{i+1}")
            ax.add_patch(circle)

        ax.plot(self.start[0], self.start[1], 'go', markersize=12, label='Start (0,0)')
        ax.plot(self.goal[0], self.goal[1], 'bo', markersize=12, label='Goal (20,20)')

        if path is not None and len(path) > 0:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            ax.plot(path_x, path_y, 'k-', color='black', linewidth=3, alpha=0.8, label='Bug0 Path')
        
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Bug0 Algorithm')
        ax.legend()

        plt.tight_layout()
        plt.savefig('bug0_path.png')
        plt.show()


class Bug0:
    def __init__(self, config_space):
        self.config_space = config_space
        self.step_size = 0.1
        self.goal_threshold = 0.4
        self.path = []
        self.all_paths = []
        self.all_times = []
        self.all_lengths = []

    def get_direction_to_goal(self, x, y) -> tuple:
        '''
        Calculate the unit vector direction from (x, y) to the goal.
        '''
        goal_x, goal_y = self.config_space.goal
        dx, dy = goal_x - x, goal_y - y
        distance = math.sqrt(dx**2 + dy**2)
        return (dx/distance, dy/distance)


    def follow_wall(self, x, y, heading=0) -> tuple:
        '''
        Follow the wall by checking for valid points in a circular arc.
        Try to find a new point by rotating around the current heading.
        '''
        for angle_offset in np.linspace(math.pi/6, 2*math.pi, 36):
            angle = heading + angle_offset
            nx = x + self.step_size * math.cos(angle)
            ny = y + self.step_size * math.sin(angle)

            if self.config_space.check_valid_point(nx, ny):
                return nx, ny, angle

        return x, y, heading 


    def calculate_path_length(self) -> float:
        '''
        Calculate the total length of the path taken by Bug0.
        '''
        length = 0.0
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i-1]
            x2, y2 = self.path[i]
            length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return length


    def run(self, n_runs=20) -> list:
        '''
        Run the Bug0 algorithm for a specified number of runs and return the best path found.
        '''
        total_time = 0.0
        total_path_length = 0.0

        for i in range(n_runs):
            self.path = []
            x, y = self.config_space.start
            heading = 0
            self.path.append((x, y))

            start_time = time.time()

            while self.config_space.distance_to_goal(x, y) > self.goal_threshold:
                dx, dy = self.get_direction_to_goal(x, y)
                heading = math.atan2(dy, dx)
                nx, ny = x + self.step_size * dx, y + self.step_size * dy

                if not self.config_space.check_valid_point(nx, ny):
                    nx, ny, heading = self.follow_wall(x, y, heading)

                x, y = nx, ny
                self.path.append((x, y))

                if len(self.path) > 5000:
                    print("Stopped due to path length limit")
                    break
            
            end_time = time.time()
            run_time = end_time - start_time
            path_length = self.calculate_path_length()

            self.all_paths.append(self.path)
            self.all_lengths.append(path_length)

            total_time += run_time
            total_path_length += path_length

            print(f"Run {i+1}: Path length = {path_length: }, Time = {run_time: } s")

        avg_time = total_time / n_runs
        avg_path_length = total_path_length / n_runs

        print(f"\nAverage computational time over {n_runs} runs: {avg_time: } s")
        print(f"Average path length over {n_runs} runs: {avg_path_length: } units")
        
        least_len_idx = self.all_lengths.index(min(self.all_lengths))
        path = self.all_paths[least_len_idx]
        
        return path


if __name__ == "__main__":
    config_space = ConfigurationSpace()
    bug0 = Bug0(config_space)

    path = bug0.run()
    config_space.visualize(path)
