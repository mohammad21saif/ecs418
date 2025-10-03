import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class HerdingEnv:
    def __init__(self, grid_size, n_dogs, n_sheeps, field_size, goal_center, goal_radius, dog_pos, sheep_pos, desired_dog_rad, init_phi, offset_dist):
        self.n_dogs = n_dogs
        self.n_sheeps = n_sheeps

        self.field_size = field_size # for repulsive potential field of sheeps

        self.grid_size = grid_size
        self.goal_center = goal_center
        self.goal_radius = goal_radius

        self.dog_init_pos = dog_pos
        self.sheep_init_pos = sheep_pos
        self.sheep_mean = [np.mean(sheep_pos, axis=0)]
        self.desired_dog_rad = desired_dog_rad
        self.init_phi = init_phi # initial heading of herd
        self.offset_dist = offset_dist # for putting virtual point in front of the herd

        self.dog_path = [[] for _ in range(n_dogs)]
        self.sheep_path = [[] for _ in range(n_sheeps)]

        dt = 0.1 # control timestep
        kp, kd, k_rho, k_alpha, k_beta = 1.0, 0.1, 1.0, 2.0, -0.5 # unicycle controller gains



    def reset(self):
        pass
        

    def step(self):
        pass


    def plot(self):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

        goal_circle = plt.Circle(self.goal_center, self.goal_radius, color='green', alpha=0.3, label='Goal Area')
        ax.add_artist(goal_circle)

        ax.scatter(self.dog_init_pos[:, 0], self.dog_init_pos[:, 1], c='blue', marker='o', label='Dogs')
 
        ax.scatter(self.sheep_init_pos[:, 0], self.sheep_init_pos[:, 1], c='red', marker='o', label='Sheep')

        # Add dog and sheep paths later when available

        ax.set_title('Herding Environment')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.legend()

        plt.tight_layout()
        plt.savefig('herding_env.png', dpi=150)
        plt.show(block=True)  # ensures window stays open

    def animate(self):
        pass


def main():
    grid_size = (100, 100)
    n_dogs = 3
    n_sheeps = 10
    field_size = (100, 100)
    goal_center = (80, 80)
    goal_radius = 10
    dog_pos = np.array([[20, 20], [20, 25], [25, 20]])
    sheep_pos = np.random.rand(n_sheeps, 2) * 20 + 40
    desired_dog_rad = 5
    init_phi = np.pi / 4
    offset_dist = 3

    env = HerdingEnv(grid_size, n_dogs, n_sheeps, field_size, goal_center, goal_radius, dog_pos, sheep_pos, desired_dog_rad, init_phi, offset_dist)

    env.plot()
    # env.animate()

if __name__ == "__main__":
    main()

    
