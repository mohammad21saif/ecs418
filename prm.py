import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import math
from enum import Enum
import warnings
import time
import networkx as nx  

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


    def visualize(self, path=None, samples=None, roadmap=None) -> None:
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
            ax.plot(path_x, path_y, 'k-', color='black', linewidth=3, alpha=0.8, label='PRM Path')

        if samples is not None:
            xs, ys = zip(*samples)
            ax.plot(xs, ys, 'y.', alpha=0.5, markersize=5, label="Samples")

        if roadmap is not None:
            for (u, v) in roadmap.edges:
                x1, y1 = roadmap.nodes[u]['pos']
                x2, y2 = roadmap.nodes[v]['pos']
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3)
        
        ax.grid(True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('PRM Algorithm')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('prm_path.png')
        plt.show()



class PRM:
    def __init__(self, config_space, n_samples, k) -> None:
        self.config_space = config_space
        self.n_samples = n_samples
        self.k = k
        self.samples = []
        self.roadmap = nx.Graph()
        self.path = []
        self.all_paths = []
        self.all_lengths = []
        self.all_times = []
        self.all_roadmaps = []
        self.all_samples = []

    def sample_points(self) -> list:
        '''
        Sample random points in the configuration space.
        '''
        samples = []
        while len(samples) < self.n_samples:
            x = np.random.uniform(self.config_space.x_min, self.config_space.x_max)
            y = np.random.uniform(self.config_space.y_min, self.config_space.y_max)

            if self.config_space.check_valid_point(x, y):
                samples.append((x, y))
        return samples
    

    def build_roadmap(self) -> None:
        '''
        Build the PRM roadmap by connecting sampled points.'''
        self.samples = self.sample_points()
        nodes = [(i, pos) for i, pos in enumerate(self.samples)]
        self.roadmap.add_nodes_from([(i, {"pos": pos}) for i, pos in nodes])

        for i, p1 in nodes:
            distances = []
            for j, p2 in nodes:
                if i != j:
                    distances.append((math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), j, p2))
            distances.sort(key=lambda x: x[0])
            for _, j, p2 in distances[:self.k]:
                if self.config_space.check_edge_collision(p1, p2):
                    self.roadmap.add_edge(i, j, weight=math.dist(p1, p2))


    def find_nearest_point(self, point) -> tuple:
        '''
        Find the nearest sampled point to the given point.
        '''
        return min(self.samples, key=lambda s: math.sqrt((s[0] - point[0])**2 + (s[1] - point[1])**2))


    def dijkstra(self) -> list:
        '''
        Find the shortest path from start to goal using Dijkstra's algorithm.
        '''
        near_start = self.find_nearest_point(self.config_space.start)
        near_goal = self.find_nearest_point(self.config_space.goal)

        start_id = len(self.samples)
        goal_id = len(self.samples) + 1

        self.roadmap.add_node(start_id, pos=self.config_space.start)
        self.roadmap.add_node(goal_id, pos=self.config_space.goal)

        for node, pos in [(start_id, self.config_space.start), (goal_id, self.config_space.goal)]:
            dists = [(math.dist(pos, s), idx, s) for idx, s in enumerate(self.samples)]
            dists.sort(key=lambda x: x[0])
            for _, j, p2 in dists[:self.k]:
                if self.config_space.check_edge_collision(pos, p2):
                    self.roadmap.add_edge(node, j, weight=math.sqrt((pos[0] - p2[0])**2 + (pos[1] - p2[1])**2))

        path_ids = nx.shortest_path(self.roadmap, source=start_id, target=goal_id, weight="weight")
        self.path = [self.roadmap.nodes[i]['pos'] for i in path_ids]
        return self.path


    def run(self, n_runs=20) -> None:
        '''
        Run the PRM algorithm for a specified number of runs and return the best path found.
        '''
        total_time = 0.0
        total_path_length = 0.0

        for run_idx in range(n_runs):
            start_time = time.time()
            self.path = []
            self.samples = []
            self.roadmap = nx.Graph()

            self.build_roadmap()
            self.path = self.dijkstra()

            end_time = time.time()
            runtime = end_time - start_time
            total_time += runtime

            path_length = 0
            for i in range(1, len(self.path)):
                path_length += math.dist(self.path[i-1], self.path[i])
            total_path_length += path_length

            self.all_paths.append(self.path)
            self.all_lengths.append(path_length)
            self.all_roadmaps.append(self.roadmap)
            self.all_samples.append(self.samples)

            print(f"Run {run_idx+1}: Path length = {path_length: }, Time = {runtime: } s")

        avg_time = total_time / n_runs
        avg_path_length = total_path_length / n_runs

        print(f"\nAverage computational time over {n_runs} runs: {avg_time: } seconds")
        print(f"Average path length over {n_runs} runs: {avg_path_length: } units")

        least_len_idx = self.all_lengths.index(min(self.all_lengths))
        path = self.all_paths[least_len_idx]
        sample = self.all_samples[least_len_idx]
        road = self.all_roadmaps[least_len_idx]

        return path, sample, road


if __name__ == '__main__':
    config_space = ConfigurationSpace()
    prm = PRM(config_space, n_samples=1000, k=10)

    path, samples, roadmap = prm.run()
    config_space.visualize(path, samples, roadmap)