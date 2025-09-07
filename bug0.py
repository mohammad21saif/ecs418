import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import math
from enum import Enum


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

    
    def check_collision(self, x, y, margin=0.1):
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


    def visualize(self, path=None):
        fig, ax = plt.subplots(1, 1, figsize=(10,10))

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')

        for i, (obsx, obsy, r) in enumerate(self.obstacles):
            circle = Circle((obsx, obsy), r, color='red', alpha=0.7, label=f"Obstacle{i+1}")
            ax.add_patch(circle)

        ax.plot(self.start[0], self.start[1], 'go', markersize=12, label='Start (1,1)')
        ax.plot(self.goal[0], self.goal[1], 'bo', markersize=12, label='Goal (20,20)')

        if path is not None and len(path) > 0:
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
        ax.plot(path_x, path_y, 'k-', color='black', linewidth=3, alpha=0.8, label='Bug0 Path')
        
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Bug2 Algorithm')
        ax.legend()

        plt.tight_layout()
        plt.show()



class Bug0:
    def __init__(self, config_space):
        self.config_space = config_space
        self.step_size = 0.1
        self.goal_threshold = 0.4
        self.path = []

    def get_direction_to_goal(self, x, y):
        goal_x, goal_y = self.config_space.goal
        dx, dy = goal_x - x, goal_y - y
        distance = math.sqrt(dx**2 + dy**2)
        return (dx/distance, dy/distance)

    
    def follow_wall(self, x, y, heading=0):
        for angle_offset in np.linspace(math.pi/6, 2*math.pi, 36):
            angle = heading + angle_offset
            nx = x + self.step_size * math.cos(angle)
            ny = y + self.step_size * math.sin(angle)

            if self.config_space.check_valid_point(nx, ny):
                return nx, ny, angle

        return x, y, heading 


    def run(self):
        x, y = self.config_space.start
        heading = 0 
        self.path.append((x, y))

        while self.config_space.distance_to_goal(x, y) > self.goal_threshold:
            dx, dy = self.get_direction_to_goal(x, y)
            heading = math.atan2(dy, dx)  
            nx, ny = x + self.step_size * dx, y + self.step_size * dy

            if not self.config_space.check_valid_point(nx, ny):
                nx, ny, heading = self.follow_wall(x, y, heading)

            x, y = nx, ny
            self.path.append((x, y))

            if len(self.path) > 5000:
                print("Stopped: too many steps")
                break
            
        print("Reached goal region!")
        return self.path


if __name__ == "__main__":
    config_space = ConfigurationSpace()
    bug0 = Bug0(config_space)

    path = bug0.run()
    config_space.visualize(path)

    