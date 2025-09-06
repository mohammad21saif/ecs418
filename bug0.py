import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import math


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

    
    def check_collision(self, x, y, margin):
        for obs_x, obs_y, radius in self.obstacles:
            distance = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            if distance <= radius + margin:
                return True
        return False


    def check_valid_point(self, x, y):
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return False
        return not self.check_collision(x, y)


    def distance_to_goal(self, x, y):
        return math.sqrt((x - self.goal[0])**2 + (y - self.goal[1])**2)


    def visualize(self, path=None, hit_points=None, leave_points=None):
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')

        for i, (obsx, obsy, r) in enumerate(self.obstacles):
            circle = Circle((obsx, obsy), r, color='red', alpha=0.7, label=f"Obstacle{i+1}")
            ax.add_patch(circle)

        ax.plot(self.start[0], self.start[1], 'go', markersize=12, label='Start (1,1)')
        ax.plot(self.goal[0], self.goal[1], 'bo', markersize=12, label='Goal (20,20)')

        ax.plot([self.start[0], self.start[1]], [self.goal[0], self.goal[1]], 'b--', color='green', alpha=0.5, linewidth=2, label='Start to Goal')

        if path is not None and len(path) > 0:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
        ax.plot(path_x, path_y, 'k-', color='black', linewidth=3, alpha=0.8, label='Bug0 Path')
        
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Bug2 Algorithm')

        plt.tight_layout()
        plt.show()
