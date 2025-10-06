import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
num_sheep = 1
num_dogs = 2
grid_size = 15.0

dt = 0.05  # timestep

# Sheep params
sheep_max_speed = 0.7
sheep_noise = 0.01
K_dog = 1.0
eps = 1e-6

cohesion_radius = 2.0
cohesion_strength = 0.35
separation_radius = 0.35
separation_strength = 0.8

# Dog (unicycle) params
dog_max_speed = 1.5
dog_max_angular_speed = 4.5
offset_distance = 0.6     # c in the derivation
formation_radius = 3.0
collect_radius = 2.0
behind_distance = 3.5

drive_advance_speed = 0.25

# switching & goal
goal = np.array([13.0, 13.0])
goal_threshold = 0.6
formation_tolerance = 10.0
herd_dispersion_threshold = 1.05

np.random.seed(0)

# ---------- Initialization ----------
sheep_positions = np.random.normal(loc=grid_size/2, scale=0.6, size=(num_sheep,2))
dogs_states = np.array([[2.0 + 0.3*i, 2.0, 0.0] for i in range(num_dogs)])  # x,y,theta

# ---------- Helpers ----------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def herd_center(positions):
    return np.mean(positions, axis=0)

def herd_dispersion(positions):
    c = herd_center(positions)
    return np.mean(np.linalg.norm(positions - c, axis=1))

def formation_error(dog_xy, targets):
    return np.mean(np.linalg.norm(dog_xy - targets, axis=1))

# ---------- Correct point-offset controller ----------
def unicycle_track_point_with_feedforward(state, target_point, target_velocity, offset_c,
                                         v_max, omega_max, Kp_point, dt):
    x, y, theta = state
    p_x = x + offset_c * np.cos(theta)
    p_y = y + offset_c * np.sin(theta)
    err = np.array([target_point[0] - p_x, target_point[1] - p_y])
    p_dot_des = Kp_point * err + target_velocity
    v = np.cos(theta) * p_dot_des[0] + np.sin(theta) * p_dot_des[1]
    omega = (-np.sin(theta) * p_dot_des[0] + np.cos(theta) * p_dot_des[1]) / max(offset_c, 1e-8)
    v = np.clip(v, -v_max, v_max)
    omega = np.clip(omega, -omega_max, omega_max)
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = wrap_angle(theta + omega * dt)
    return np.array([x_new, y_new, theta_new]), v, omega

# ---------- Target generation ----------
def compute_arc_targets(arc_center, goal, n_dogs, radius):
    v = goal - arc_center
    ang_to_goal = np.arctan2(v[1], v[0])
    arc_center_angle = wrap_angle(ang_to_goal + np.pi)
    arc_span = np.pi * 0.9
    angles = arc_center_angle + np.linspace(-arc_span/2, arc_span/2, n_dogs)
    targets = np.array([arc_center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
    return targets

def compute_collect_targets(sheep_positions, n_dogs, radius):
    c = herd_center(sheep_positions)
    angles = np.linspace(0, 2*np.pi, n_dogs, endpoint=False)
    targets = np.array([c + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
    return targets

# ---------- Sheep update ----------
def update_sheep_positions_paper(sheep_positions, dog_positions, dt):
    n = len(sheep_positions)
    new_pos = sheep_positions.copy()
    for i in range(n):
        s = sheep_positions[i].copy()
        # dog repulsion term
        s_dot_dogs = np.zeros(2)
        for d in dog_positions:
            diff = s - d
            dist = np.linalg.norm(diff)
            denom = max(dist, eps)**3
            s_dot_dogs += diff / denom
        s_dot_dogs *= K_dog
        # cohesion
        coh_vec = np.zeros(2)
        count = 0
        for j in range(n):
            if j == i: continue
            other = sheep_positions[j]
            if np.linalg.norm(other - s) < cohesion_radius:
                coh_vec += other
                count += 1
        if count > 0:
            coh_center = coh_vec / count
            coh_dir = coh_center - s
        else:
            coh_dir = np.zeros(2)
        # separation
        sep = np.zeros(2)
        for j in range(n):
            if j == i: continue
            other = sheep_positions[j]
            diff = other - s
            dist = np.linalg.norm(diff)
            if dist < separation_radius and dist > 1e-9:
                sep -= separation_strength * (diff / dist) * ((separation_radius - dist) / separation_radius)
        # total velocity
        vel = s_dot_dogs + cohesion_strength * coh_dir + sep
        vel += np.random.normal(scale=sheep_noise, size=2)
        speed = np.linalg.norm(vel)
        if speed > 1e-9:
            vel = vel / speed * min(speed, sheep_max_speed)
        new_pos[i] = np.clip(s + vel * dt, 0, grid_size)
    return new_pos

# ---------- Simulation ----------
plt.ion()
fig = plt.figure(figsize=(8,8))

phase = "COLLECT"
drive_arc_center = None
last_phase = phase
Kp_point = 1.4  # point feedback gain

# --- Added for per-dog feedforward velocity ---
old_drive_arc_center = None
old_dog_targets = None

for step in range(3000):
    c = herd_center(sheep_positions)
    disp = herd_dispersion(sheep_positions)
    dogs_xy = dogs_states[:,:2]

    # goal reached check
    if np.linalg.norm(c - goal) < goal_threshold:
        print("Goal reached at step", step, "centroid", c)
        break

    behind_vec = c - goal
    norm_b = np.linalg.norm(behind_vec)
    if norm_b < 1e-9:
        behind_unit = np.array([1.0, 0.0])
    else:
        behind_unit = behind_vec / norm_b
    proposed_arc_center = c + behind_distance * behind_unit
    proposed_drive_targets = compute_arc_targets(proposed_arc_center, goal, num_dogs, formation_radius)
    form_err = formation_error(dogs_xy, proposed_drive_targets)

    # switching logic
    if disp > herd_dispersion_threshold:
        phase = "COLLECT"
    else:
        if form_err < formation_tolerance:
            phase = "DRIVE"
        else:
            phase = "COLLECT"

    if phase == "DRIVE" and last_phase != "DRIVE":
        u = goal - c
        un = u / (np.linalg.norm(u) + 1e-9)
        drive_arc_center = c - behind_distance * un
        old_drive_arc_center = drive_arc_center.copy()
        old_dog_targets = compute_arc_targets(drive_arc_center, goal, num_dogs, formation_radius)
    last_phase = phase

    if phase == "DRIVE":
        u = goal - c
        u_norm = np.linalg.norm(u)
        u_unit = u / (u_norm + 1e-9)
        old_drive_arc_center = drive_arc_center.copy()
        drive_arc_center = drive_arc_center + drive_advance_speed * u_unit * dt
        # get old and new dog targets
        old_dog_targets = compute_arc_targets(old_drive_arc_center, goal, num_dogs, formation_radius)
        dog_targets = compute_arc_targets(drive_arc_center, goal, num_dogs, formation_radius)
        target_velocities = (dog_targets - old_dog_targets) / dt  # per-dog velocity
    else:
        dog_targets = compute_collect_targets(sheep_positions, num_dogs, collect_radius)
        target_velocities = np.zeros_like(dog_targets)

    # Move dogs (main fix below: per-dog target_velocity)
    for i in range(num_dogs):
        state = dogs_states[i]
        target = dog_targets[i]
        new_state, v_cmd, omega_cmd = unicycle_track_point_with_feedforward(
            state, target, target_velocities[i], offset_distance,
            dog_max_speed, dog_max_angular_speed, Kp_point, dt
        )
        new_state[0] = np.clip(new_state[0], 0, grid_size)
        new_state[1] = np.clip(new_state[1], 0, grid_size)
        dogs_states[i] = new_state

    # Update sheep using paper repulsion; preserve centroid in COLLECT
    dogs_positions_xy = dogs_states[:,:2]
    if phase != "DRIVE":
        c_before = c.copy()
        new_sheep = update_sheep_positions_paper(sheep_positions, dogs_positions_xy, dt)
        c_after = herd_center(new_sheep)
        shift = c_before - c_after
        new_sheep += shift
        sheep_positions = np.clip(new_sheep, 0, grid_size)
    else:
        sheep_positions = update_sheep_positions_paper(sheep_positions, dogs_positions_xy, dt)

    # plot
    plt.clf()
    plt.scatter(sheep_positions[:,0], sheep_positions[:,1], c='orange', label='Sheep')
    plt.scatter(dogs_states[:,0], dogs_states[:,1], c='blue', label='Dogs')
    for i in range(num_dogs):
        x,y,th = dogs_states[i]
        plt.arrow(x, y, 0.35*np.cos(th), 0.35*np.sin(th), head_width=0.12, head_length=0.12, color='blue')
    plt.scatter(dog_targets[:,0], dog_targets[:,1], c='red', marker='x', label='Dog targets')
    plt.scatter([c[0]],[c[1]], c='green', marker='*', s=100, label='Herd center')
    plt.scatter([goal[0]],[goal[1]], c='purple', marker='P', s=120, label='Goal')
    plt.xlim(0, grid_size); plt.ylim(0, grid_size)
    plt.title(f"Step:{step} Phase:{phase} Disp:{disp:.2f} FormErr:{form_err:.2f}")
    plt.legend(loc='upper left')
    plt.pause(0.01)

plt.ioff()
plt.show()
