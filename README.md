# Cart-Pole Optimal Control

**SES598 Space Robotics and AI | Arizona State University**  
**Kacy | February 2026**

> System formalism based on: https://underactuated.mit.edu/acrobot.html#cart_pole

---

## Overview

This assignment implements and analyzes an LQR controller for cart-pole stabilization under continuous earthquake disturbances, with a DQN reinforcement learning controller implemented for extra credit comparison. The system runs in ROS2 Humble with Gazebo Garden simulation on Ubuntu 22.04 (WSL2).

The cart-pole consists of an inverted pendulum mounted on a sliding cart. The goal is to keep the pole upright while preventing the cart from exceeding its physical limits of +/-2.5m, all while a continuous earthquake force generator applies realistic seismic disturbances.

---

## System Description

### Physical Parameters

| Parameter | Value |
|-----------|-------|
| Cart mass (M) | 1.0 kg |
| Pole mass (m) | 1.0 kg |
| Pole length (L) | 1.0 m |
| Cart limits | +/-2.5 m |
| Control rate | 50 Hz |

### State Vector

```
x = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
```

### Linearized Dynamics

Linearizing around the upright equilibrium (theta = 0) gives the state-space form `x_dot = Ax + Bu`, where the matrices are derived from the Euler-Lagrange equations of the cart-pole system:

```
A = [[0,  1,       0,          0      ],
     [0,  0,    mg/M,          0      ],
     [0,  0,       0,          1      ],
     [0,  0, (M+m)g/ML,        0      ]]

B = [0,  1/M,  0,  -1/(ML)]^T
```

Substituting M = m = L = 1 and g = 9.81 m/s^2 gives the numerical matrices used in the controller.

### Earthquake Disturbance Model

The disturbance generator produces earthquake-like forces via superposition of sine waves:

```
F_eq(t) = sum_i [ a_i * sin(2*pi*f_i*t + phi_i) ] + noise

where:
  a_i   ~ Uniform(0.8, 1.2) * 15 N
  f_i   ~ Uniform(0.5, 4.0) Hz
  phi_i ~ Uniform(0, 2*pi)
  noise ~ Normal(0, 1.5)
```

Five waves superimposed can constructively interfere to produce peaks up to approximately +/-75 N — five times the base amplitude of 15 N. This makes earthquake rejection the dominant challenge for both controllers.

### Force Architecture

The earthquake generator publishes to `/earthquake_force` for visualization and to `/model/cart_pole/joint/cart_to_base/cmd_force` for Gazebo actuation. The LQR controller subscribes to `/earthquake_force`, computes its control force `u = -Kx`, and publishes the combined total `u + F_eq` to the same Gazebo topic. This ensures the earthquake acts as a true external disturbance that the LQR must overcome, rather than the two publishers racing to overwrite each other.

---

## LQR Controller

### Theory

The Linear Quadratic Regulator minimizes the infinite-horizon quadratic cost:

```
J = integral_0^inf ( x^T Q x + u^T R u ) dt
```

The matrix Q penalizes state deviations and R penalizes control effort. The optimal gain is found by solving the Algebraic Riccati Equation (ARE):

```
A^T P + P A - P B R^-1 B^T P + Q = 0
```

Once P is found, the optimal gain matrix and control law are:

```
K = R^-1 B^T P
u = -K x
```

This is the mathematically optimal linear state feedback for the given Q and R. The choice of Q and R entirely determines the performance trade-off, which is why tuning them carefully matters.

**Connection to eigenvalues:** The ARE ensures all eigenvalues of the closed-loop system `(A - BK)` are in the left half-plane (stable). The Q and R matrices indirectly control how far left these eigenvalues are placed, determining response speed versus control effort trade-off.

### Default Parameter Analysis

The default parameters provided are:

```python
Q = np.diag([1.0, 1.0, 10.0, 10.0])  # [x, x_dot, theta, theta_dot]
R = np.array([[0.1]])
```

The pole angle and angular velocity are weighted 10x higher than cart states, which correctly prioritizes keeping the pole upright. However, testing showed these defaults cause persistent cart drift until the +/-2.5m limit is hit at approximately 7 seconds. This motivated a systematic parameter search.

### Bayesian Optimization

Rather than manually tuning the five parameters of Q and R, Bayesian optimization was applied using the same technique from the boustrophedon navigator assignment for PD controller tuning. A Gaussian Process surrogate model with a Matern kernel (nu = 2.5) was fit to observed performance scores, with Expected Improvement as the acquisition function.

**Search space (widened from initial run to explore higher Q_theta and Q_theta_dot ranges):**

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|-------------|
| Q_x | 1.0 | 50.0 |
| Q_x_dot | 1.0 | 50.0 |
| Q_theta | 10.0 | 200.0 |
| Q_theta_dot | 10.0 | 200.0 |
| R | 0.1 | 10.0 |

**Objective function** (simulated via Euler integration at 20ms timesteps, averaged over 3 runs to reduce earthquake randomness noise):

```
score = 2*t_stable - 5*x_max - 0.5*theta_max - 0.1*|u|_avg - 3*x_rms
```

The optimization ran for 10 random initialization samples followed by 75 Bayesian iterations (85 total evaluations). The theoretical maximum score is approximately 240 (surviving the full 120s simulation with zero penalties).

**Optimal parameters found:**

```python
Q = np.diag([31.513, 49.947, 60.977, 186.931])
R = np.array([[0.1648]])
```

Best score: **238.27 / 240 theoretical maximum** — near-perfect performance in simulation.

**Parameter importance findings:**

- **Q_thetadot (186.9) dominates** the optimal solution — heavily penalizing angular velocity prevents the rotational momentum buildup that causes sudden instability under large earthquake spikes.
- **R is very low (0.165)** — the controller is permitted aggressive force responses, which is necessary to counter earthquake peaks of ±75N.
- **R showed the strongest negative correlation (-0.63)** with score, confirming that high control cost is the single biggest performance limiter under large disturbances.
- The best solution was found at **iteration 13** and held through all remaining 62 iterations, indicating well-converged search.

**Resulting LQR gain matrix:**

```
K = [[-13.828, -24.207, -320.677, -69.496]]
```

The large -320.7 weight on pole angle reflects the high Q_theta value, making the controller highly sensitive to any pole deviation.

### LQR Performance Results

| Metric | Assignment Defaults | Bayesian Optimized |
|--------|-------------------|---------------------|
| Q matrix | [1, 1, 10, 10] | [31.51, 49.95, 60.98, 186.93] |
| R value | 0.1 | 0.165 |
| Stable Duration | ~7.0 s | **~115 s** |
| Max Cart Displacement | 0.6+ m | ~0.05 m |
| Pole Angle Range | ±5° then failure | ±5° for 115s |
| Failure Mode | Cart drift to limit | Large earthquake spike |

---

## Results

### Default LQR Parameters

![Default LQR Performance](iteration_1.png)

With the assignment default parameters `Q = diag([1, 1, 10, 10])` and `R = [[0.1]]`, the pole angle is well controlled (near 0 degrees throughout) but the cart drifts steadily to 0.6m before sudden failure at approximately 7 seconds. This motivated the Bayesian optimization run.

### Bayesian Optimization

The optimizer ran 10 random samples followed by 75 Bayesian iterations. The best score (238.27) was found at iteration 13 and held through all remaining iterations.

![LQR Bayesian Optimization Results](figures/lqr_top_v2.png)

The four plots show:
- **Optimization Convergence (top left):** scores fluctuate between 230 and 238 across 85 evaluations, with the best score of 238.27 found at iteration 13
- **Best Score Progress (top right):** rapid improvement in the first 25 iterations then complete convergence — the solution was found early and never improved upon
- **Parameter Importance (bottom left):** Q_x and Q_xdot positively correlated with score; R strongly negatively correlated (-0.63), meaning high control cost is the single biggest performance limiter
- **Optimal Q Weights (bottom right):** Q_thetadot (186.9) dominates by a wide margin, with Q_theta (61.0), Q_xdot (50.0), and Q_x (31.5) providing secondary stabilization

Selected optimization log:

```
Phase 1: Random exploration (10 samples)
  Sample 1:  Q=[19.35,47.59,149.08,123.75], R=1.645 -> Score=235.32
  Sample 10: Q=[34.91,43.74,191.19,181.05], R=1.922 -> Score=236.34

Phase 2: Bayesian optimization (75 iterations)
  Iter  1: Q=[37.92,8.29,181.04,83.38],   R=0.253 -> Score=237.33 (Best=237.33)
  Iter 10: Q=[40.61,24.28,176.18,37.51],  R=0.160 -> Score=237.40 (Best=237.40)
  Iter 13: Q=[31.51,49.95,60.98,186.93],  R=0.165 -> Score=238.27 (Best=238.27)  <-- BEST
  Iter 75: Q=[20.73,37.03,165.58,42.55],  R=1.583 -> Score=234.91 (Best=238.27)

OPTIMAL: Q = diag([31.513, 49.947, 60.977, 186.931]),  R = [[0.1648]],  Score = 238.27
```

### Optimized LQR in ROS2/Gazebo

Running with the Bayesian-optimized parameters in the full ROS2/Gazebo simulation with properly combined forces:

![Optimized LQR Performance](figures/lqr_v3.png)

The cart stays within approximately ±0.05m of center for the entire run — essentially perfect position control. The pole angle remains within ±5° throughout. Both the cart and pole maintain stability until a large earthquake spike at approximately 115 seconds overwhelms the controller and drives the cart past the +2.5m limit. The control forces reach up to ±100N as the controller fights ±75N earthquake peaks.

This is the same failure mode seen with the default parameters (sudden instability from earthquake spike rather than gradual drift), just delayed by more than 100 seconds.

### DQN Training

The improved training script fixed three issues present in the original: the `force_mag` accumulation bug that caused earthquake forces to grow unbounded, inadequate action force magnitude relative to earthquake peaks, and a reward function with poor gradients near failure. Training converged in 120 episodes:

```
Episode   10 | Avg Reward (100ep):    ~20   | ε: 1.00
Episode   30 | Avg Reward (100ep):    ~100  | ε: 0.05  (epsilon reached minimum)
Episode  110 | Avg Reward (100ep):   415.17 | Steps:  187 | ε: 0.0500
Episode  120 | Avg Reward (100ep):   474.90 | Steps:  183 | ε: 0.0500

Solved at episode 120 (threshold: 450)
Best average reward: 474.90
```

![DQN Training Progress](cart_pole_optimal_control/dqn/dqn_train_v3.png)

Epsilon decayed to 0.05 by episode 30 (vs episode 500+ in the original due to `decay=0.9999` per step). Once exploitation began, reward climbed steadily from ~100 to 474 by episode 120. The agent solved the task 30 episodes faster than the previous implementation.

### DQN Evaluation in ROS2/Gazebo

![DQN Evaluation](cart_pole_optimal_control/dqn/dqn_lqr_v2.png)

The DQN lasted approximately 47 seconds before the cart hit the +2.5m limit. The pole angle was well controlled (within ±5°) throughout the run, matching LQR pole performance. However, the cart drifted steadily from center to the boundary due to the discrete action constraint. The control force plot shows bang-bang behavior — the DQN can only apply exactly +15N or -15N, producing square wave switching. This is fundamentally different from LQR's smooth proportional control and explains the cart positioning failure.

---

## LQR vs DQN Comparison

| Metric | LQR (Optimized) | DQN |
|--------|----------------|-----|
| Stable Duration | **~115 s** | ~47 s |
| Max Cart Displacement | **~0.05 m** | 2.5 m (limit hit) |
| Pole Angle Control | ±5° | ±5° |
| Control Type | Continuous, proportional | Discrete, bang-bang |
| Avg Control Effort | Proportional to error | Fixed ±15N |
| Training Required | None (analytical) | 120 episodes |
| Parameter Count | 5 (Q + R values) | ~8,000 (network weights) |
| Interpretability | High (physical meaning) | Low (black box) |
| Disturbance Awareness | Reactive only | Earthquake in state vector |
| Failure Mode | Earthquake spike at 115s | Cart drift at 47s |

### Discussion

Both controllers kept the pole upright equally well — the difference is entirely in cart position control, which comes down to the fundamental difference between continuous and discrete action spaces.

**LQR advantages:** Continuous force output enables precise cart positioning. The Riccati equation provides mathematical optimality guarantees for the linearized system. Each element of K has direct physical meaning as sensitivity to a state variable. Zero training time — analytical solution from Q and R matrices. Bayesian optimization of Q and R using a GP surrogate found parameters that improved stability from 7s to 115s (a 16x improvement).

**DQN advantages:** Direct earthquake force awareness through augmented state — the LQR is purely reactive to resulting state, while the DQN can in principle anticipate disturbances. Model-free learning requires no knowledge of system dynamics. Can handle nonlinearities beyond the linearization region.

**DQN limitations:** Discrete action space (±15N only) fundamentally prevents fine cart positioning. Even with 40N action force, earthquake peaks of ±75N often exceed control authority. A continuous-action variant such as DDPG or SAC would be the natural next step to close the performance gap.

**Key insight:** LQR is the more appropriate controller for this problem because the system dynamics are well understood, the linearization is accurate near the upright equilibrium, and the continuous action space is critical for position control under large disturbances. DQN becomes competitive in high-dimensional, nonlinear, or partially observable systems where analytical control design is intractable.

---

## DQN Controller (Extra Credit)

### Architecture

The DQN uses a fully connected neural network to approximate Q(s, a):

```
Input (5) -> FC1 (64, ReLU) -> FC2 (64, ReLU) -> Output (2, Linear)
```

The state is augmented with normalized earthquake force:

```
s = [cart_pos, cart_vel, pole_angle, pole_ang_vel, F_eq / (15 * 5)]
```

Normalization divides by the theoretical peak (15N base × 5 waves = 75N) to keep inputs in a consistent range. The two outputs correspond to push left (-40N) or push right (+40N). Action force was increased from gym's default 10N to 40N to give the agent meaningful authority against ±75N earthquake peaks.

Implementation details:
- Experience replay buffer: 100,000 transitions
- Batch size: 64
- Target network updated every 10 episodes
- Epsilon decay: 1.0 to 0.05 at rate 0.995 per step (reaches minimum ~episode 30)

### Reward Function

```
r(s, done) = r_alive + r_pole + r_cart + r_vel

r_alive = 1.0
r_pole  = 2 * cos(theta)                      (smooth angle reward, max 2.0 at theta=0)
r_cart  = exp(-0.5 * (x / 2.4)^2)            (Gaussian centering, max 1.0 at center)
r_vel   = -0.01*|x_dot| - 0.005*|theta_dot|  (velocity penalty)
r(done) = -10.0                               (termination penalty)
```

The cosine pole reward provides smooth gradients near vertical. The Gaussian cart reward emphasizes centering without harshly penalizing moderate displacements. The heavy termination penalty discourages premature failure.

### Training

Training used gymnasium's CartPole-v1 environment for speed — thousands of episodes in minutes versus hours for full ROS2/Gazebo simulation. The earthquake disturbance was applied as a physics-correct state perturbation at each timestep (`v += F*dt`, `x += 0.5*F*dt^2`) rather than modifying `force_mag`. A new random earthquake profile was generated each episode to prevent overfitting to a specific disturbance pattern.

---

## Running the Code

### LQR Controller

```bash
# Terminal 1
ros2 launch cart_pole_optimal_control cart_pole.launch.py

# Terminal 2
ros2 run cart_pole_optimal_control lqr_controller

# Terminal 3
ros2 run cart_pole_optimal_control earthquake_force_generator
```

### Bayesian Optimization

```bash
cd assignments/cart_pole_optimal_control
python3 tune_lqr.py
```

### DQN Training

```bash
cd cart_pole_optimal_control/dqn/
python3 dqn_train.py
```

### DQN Evaluation

```bash
# Terminal 1
ros2 launch cart_pole_optimal_control cart_pole.launch.py

# Terminal 2
cd cart_pole_optimal_control/dqn/
python3 dqn_ros2_evaluate.py

# Terminal 3
ros2 run cart_pole_optimal_control earthquake_force_generator
```

---

## Key Findings

1. **Bayesian optimization over a wider search space (Q_thetadot up to 200, R up to 10) found parameters that improved stability from ~7s to ~115s** — a 16x improvement over assignment defaults
2. **Q_thetadot (186.9) was the dominant parameter** — penalizing angular velocity heavily prevents the momentum buildup that causes sudden failure under large earthquake spikes
3. **Low R (0.165) was critical for earthquake rejection** — high control cost was the single strongest negative predictor of performance (correlation -0.63)
4. **The force architecture matters** — fixing the topic collision between earthquake generator and LQR controller (combining rather than overwriting forces) was necessary to get valid results
5. **Discrete action spaces fundamentally limit positioning performance** — DQN's bang-bang control (±40N only) could not match LQR's continuous proportional forces, despite equal pole angle control
6. **DQN solved in 120 episodes** after fixing `force_mag` accumulation bug and increasing epsilon decay rate from 0.9999 to 0.995 per step

---

## References

Tedrake, R. (2023). Underactuated Robotics. https://underactuated.mit.edu/acrobot.html#cart_pole

Acknowledgements: Aldrin Inbaraj A, Arizona State University.
