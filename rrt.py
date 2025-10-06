import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import math
from enum import Enum
import warnings
import time
import random

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


    def check_edge_collision(self, point_1, point_2, step=0.1) -> bool:
        '''
        Check if the straight line between point_1 and point_2 collides with any obstacles'''
        x1, y1 = point_1
        x2, y2 = point_2
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        steps = max(1, int(distance / step))
        for i in range(steps + 1):
            e = i / steps
            x = x1 + e * (x2 - x1)
            y = y1 + e * (y2 - y1)
            if not self.check_valid_point(x, y):
                return False
        return True


    def visualize(self, path=None, tree=None) -> None:
        '''
        Visualize the configuration space, obstacles, start and goal points, and the PRM roadmap.
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
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, 'k-', color='black', linewidth=3, alpha=0.8, label='RRT Path')

        if tree is not None:
            for parent, child in tree:
                x_tree = [parent[0], child[0]]
                y_tree = [parent[1], child[1]]
                ax.plot(x_tree, y_tree, color='gray', alpha=0.3)

        
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('RRT Algorithm')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('rrt_path.png')
        plt.show()


class RRT:
    def __init__(self, config_space, iterations, step_size, goal_sample_rate) -> None:
        self.config_space = config_space
        self.tree = []
        self.step_size = step_size
        self.iterations = iterations
        self.goal_sample_rate = goal_sample_rate
        self.path = []
        self.all_paths = []
        self.all_lengths = []
        self.all_trees = []
        self.nodes = [self.config_space.start]


    def sample_point(self) -> tuple:
        '''
        Randomly sample a point in the configuration space, with a bias towards the goal.
        '''
        if random.random() < self.goal_sample_rate:
            return self.config_space.goal
            
        x = np.random.uniform(self.config_space.x_min, self.config_space.x_max)
        y = np.random.uniform(self.config_space.y_min, self.config_space.y_max)
        return (x,y)


    def nearest_node(self, point) -> tuple:
        '''
        Find the nearest node in the tree to the given point.
        '''
        return min(self.nodes, key=lambda node: math.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2))

    
    def steer(self, point1, point2) -> tuple:
        '''
        Steer from point1 towards point2 by step_size.
        '''
        distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        if distance < self.step_size:
            return point2
        else:
            angle = math.atan2(point2[1] - point1[1], point2[0] - point1[0])
            x = point1[0] + self.step_size * math.cos(angle)
            y = point1[1] + self.step_size * math.sin(angle)
            return (x,y)


    def run(self, n_runs=1) -> list:
        '''
        Run the RRT algorithm for a specified number of runs and return the best path found.
        '''
        total_time = 0.0
        total_path_length = 0.0

        for i in range(n_runs):
            self.nodes = [self.config_space.start]
            self.tree = []
            self.path = []

            start_time = time.time()

            for _ in range(self.iterations):
                random_point = self.sample_point()
                nearest_point = self.nearest_node(random_point)
                new_node = self.steer(nearest_point, random_point)

                if self.config_space.check_edge_collision(nearest_point, new_node):
                    self.nodes.append(new_node)
                    self.tree.append((nearest_point, new_node))
                    if math.dist(new_node, self.config_space.goal) < self.step_size:
                        self.nodes.append(self.config_space.goal)
                        self.tree.append((new_node, self.config_space.goal))
                        break

            self.path = [self.config_space.goal]
            current = self.config_space.goal
            while current != self.config_space.start:
                for parent, child in reversed(self.tree):
                    if child == current:
                        self.path.append(parent)
                        current = parent
                        break

            self.path.reverse()

            end_time = time.time()
            run_time = end_time - start_time
            path_length = sum(math.dist(self.path[i], self.path[i+1]) for i in range(len(self.path)-1))

            self.all_paths.append(self.path)
            self.all_lengths.append(path_length)
            self.all_trees.append(self.tree)

            total_time += run_time
            total_path_length += path_length

            print(f"Run {i+1}: Path length = {path_length: }, Time = {run_time: } s")

        avg_time = total_time / n_runs
        avg_path_length = total_path_length / n_runs

        print(f"\nAverage computational time over {n_runs} runs: {avg_time: } s")
        print(f"Average path length over {n_runs} runs: {avg_path_length: } units")

        least_len_idx = self.all_lengths.index(min(self.all_lengths))
        path = self.all_paths[least_len_idx]
        tree = self.all_trees[least_len_idx]

        return path, tree



if __name__ == '__main__':
    config_space = ConfigurationSpace()
    rrt = RRT(config_space, iterations=15000, step_size=0.1, goal_sample_rate=0.05)

    path, tree = rrt.run()
    config_space.visualize(path, tree)