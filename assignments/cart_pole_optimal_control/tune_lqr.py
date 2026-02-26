
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

class LQRBayesianOptimizer:
    """
    Bayesian Optimization for LQR parameter tuning.
    Search space: Q diagonal values and R value
    Objective: maximize stability (minimize cost)
    """
    def __init__(self):
        # System parameters (from lqr_controller.py)
        self.M = 1.0
        self.m = 1.0
        self.L = 1.0
        self.g = 9.81

        self.A = np.array([
            [0, 1, 0, 0],
            [0, 0, (self.m * self.g) / self.M, 0],
            [0, 0, 0, 1],
            [0, 0, ((self.M + self.m) * self.g) / (self.M * self.L), 0]
        ])

        self.B = np.array([
            [0],
            [1/self.M],
            [0],
            [-1/(self.M * self.L)]
        ])

        # Search space bounds (log scale for better coverage)
        # [q_x, q_xdot, q_theta, q_thetadot, r]
        self.bounds = np.array([
            [1.0, 50.0],   # q_x: cart position weight
            [1.0, 50.0],   # q_xdot: cart velocity weight
            [10.0, 200.0],  # q_theta: pole angle weight (most important)
            [10.0, 200.0],  # q_thetadot: pole angular velocity weight
            [0.1, 10.0],   # r: control cost
        ])

        # GP model
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )

        # Storage
        self.X_observed = []  # Parameters tried
        self.y_observed = []  # Performance scores

    def compute_lqr_gain(self, q_x, q_xdot, q_theta, q_thetadot, r):
        """Compute LQR gain for given parameters."""
        Q = np.diag([q_x, q_xdot, q_theta, q_thetadot])
        R = np.array([[r]])
        try:
            P = solve_continuous_are(self.A, self.B, Q, R)
            K = np.linalg.inv(R) @ self.B.T @ P
            return K, Q, R
        except Exception:
            return None, None, None
    def simulate_performance(self, params, n_runs=3, **kwargs):
        scores = [self._single_run(params, **kwargs) for _ in range(n_runs)]
        return np.mean(scores)

    def _single_run(self, params, dt=0.02, duration=120.0,
                              earthquake_amp=15.0, freq_range=(0.5, 4.0)):
        """
        Simulate cart-pole with given LQR parameters.
        Returns a performance score (higher = better).
        """
        q_x, q_xdot, q_theta, q_thetadot, r = params
        K, Q, R = self.compute_lqr_gain(q_x, q_xdot, q_theta, q_thetadot, r)

        if K is None:
            return -1000.0  # Invalid parameters

        # Simulate system
        x = np.zeros((4, 1))  # [cart_pos, cart_vel, pole_angle, pole_vel]
        t = 0.0
        stable_duration = 0.0

        cart_positions = []
        pole_angles = []
        control_forces = []

        # Earthquake generator (matches earthquake_force_generator.py)
        n_waves = 5
        freqs = np.random.uniform(freq_range[0], freq_range[1], n_waves)
        amps = np.random.uniform(0.5, 1.5, n_waves) * earthquake_amp / n_waves
        phases = np.random.uniform(0, 2*np.pi, n_waves)

        while t < duration:
            # Generate earthquake force
            eq_force = sum(a * np.sin(2*np.pi*f*t + p)
                          for a, f, p in zip(amps, freqs, phases))
            eq_force += np.random.normal(0, earthquake_amp * 0.1)

            # Compute control
            u = float((-K @ x)[0])
            control_forces.append(abs(u))

            # Apply total force
            total_force = u + eq_force

            # Simulate one step (Euler integration)
            x_dot = self.A @ x + self.B * total_force
            x = x + x_dot * dt

            cart_positions.append(abs(x[0, 0]))
            pole_angles.append(abs(np.degrees(x[2, 0])))

            # Check termination
            if abs(x[0, 0]) > 2.5 or abs(x[2, 0]) > np.radians(45):
                break

            stable_duration = t
            t += dt

        # Compute performance score
        max_cart = max(cart_positions) if cart_positions else 2.5
        max_angle = max(pole_angles) if pole_angles else 45.0
        avg_force = np.mean(control_forces) if control_forces else 1000.0
        rms_cart = np.sqrt(np.mean(np.array(cart_positions)**2))

        # Weighted objective (matches assignment metrics)
        score = (
            stable_duration * 2.0          # Reward longer stability
            - max_cart * 5.0               # Penalize cart displacement
            - max_angle * 0.5              # Penalize pole deviation
            - avg_force * 0.1              # Penalize control effort
            - rms_cart * 3.0              # Penalize RMS cart error
        )

        return score

    def expected_improvement(self, X, xi=0.01):
        """Acquisition function: Expected Improvement."""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        mu_best = np.max(self.y_observed)
        Z = (mu - mu_best - xi) / (sigma + 1e-9)
        EI = (mu - mu_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma < 1e-10] = 0.0
        return EI.flatten()

    def suggest_next_params(self, n_candidates=1000):
        """Suggest next parameters to try using EI."""
        # Random candidates in search space
        candidates = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_candidates, len(self.bounds))
        )

        ei = self.expected_improvement(candidates)
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def optimize(self, n_initial=5, n_iterations=20):
        """Run Bayesian optimization."""
        print("=" * 60)
        print("LQR Bayesian Optimization")
        print("=" * 60)

        # Initial random exploration
        print(f"\nPhase 1: Random exploration ({n_initial} samples)")
        for i in range(n_initial):
            params = np.random.uniform(
                self.bounds[:, 0],
                self.bounds[:, 1]
            )
            score = self.simulate_performance(params)
            self.X_observed.append(params)
            self.y_observed.append(score)
            print(f"  Sample {i+1}: Q=[{params[0]:.2f},{params[1]:.2f},"
                  f"{params[2]:.2f},{params[3]:.2f}], R={params[4]:.3f} "
                  f"-> Score={score:.2f}")

        # Bayesian optimization iterations
        print(f"\nPhase 2: Bayesian optimization ({n_iterations} iterations)")
        for i in range(n_iterations):
            # Fit GP
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)

            # Suggest next params
            next_params = self.suggest_next_params()

            # Evaluate
            score = self.simulate_performance(next_params)
            self.X_observed.append(next_params)
            self.y_observed.append(score)

            best_so_far = max(self.y_observed)
            print(f"  Iter {i+1:2d}: Q=[{next_params[0]:.2f},{next_params[1]:.2f},"
                  f"{next_params[2]:.2f},{next_params[3]:.2f}], "
                  f"R={next_params[4]:.3f} -> Score={score:.2f} "
                  f"(Best={best_so_far:.2f})")

        # Get best parameters
        best_idx = np.argmax(self.y_observed)
        best_params = self.X_observed[best_idx]
        best_score = self.y_observed[best_idx]

        print("\n" + "=" * 60)
        print("OPTIMAL PARAMETERS FOUND:")
        print(f"  Q = diag([{best_params[0]:.3f}, {best_params[1]:.3f}, "
              f"{best_params[2]:.3f}, {best_params[3]:.3f}])")
        print(f"  R = [[{best_params[4]:.4f}]]")
        print(f"  Best Score: {best_score:.2f}")
        print("=" * 60)

        self.plot_optimization_results(best_params)
        return best_params

    def plot_optimization_results(self, best_params):
        """Plot optimization progress and parameter analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('LQR Bayesian Optimization Results', fontsize=14)

        # Convergence plot
        axes[0, 0].plot(self.y_observed, 'b-o', markersize=4)
        axes[0, 0].axhline(max(self.y_observed), color='r',
                           linestyle='--', label='Best score')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Performance Score')
        axes[0, 0].set_title('Optimization Convergence')
        axes[0, 0].legend()

        # Running best
        running_best = [max(self.y_observed[:i+1])
                       for i in range(len(self.y_observed))]
        axes[0, 1].plot(running_best, 'g-o', markersize=4)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Best Score So Far')
        axes[0, 1].set_title('Best Score Progress')

        # Parameter importance - Q weights
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        param_names = ['Q_x', 'Q_xdot', 'Q_theta', 'Q_thetadot', 'R']
        correlations = [np.corrcoef(X[:, i], y)[0, 1]
                       for i in range(X.shape[1])]
        colors = ['green' if c > 0 else 'red' for c in correlations]
        axes[1, 0].bar(param_names, correlations, color=colors)
        axes[1, 0].set_ylabel('Correlation with Score')
        axes[1, 0].set_title('Parameter Importance')
        axes[1, 0].axhline(0, color='black', linewidth=0.5)

        # Best parameters visualization
        best_K, best_Q, best_R = self.compute_lqr_gain(*best_params)
        categories = param_names[:-1]
        values = best_params[:-1]
        axes[1, 1].bar(categories, values, color='steelblue')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].set_title(f'Optimal Q Weights (R={best_params[4]:.4f})')

        plt.tight_layout()
        plt.savefig('lqr_optimization_results.png', dpi=150,
                    bbox_inches='tight')
        plt.show()
        print("Plot saved as 'lqr_optimization_results.png'")


if __name__ == '__main__':
    np.random.seed(42)
    optimizer = LQRBayesianOptimizer()
    best_params = optimizer.optimize(n_initial=10, n_iterations=75)

    print("\nCopy these into lqr_controller.py:")
    print(f"self.Q = np.diag([{best_params[0]:.3f}, {best_params[1]:.3f}, "
          f"{best_params[2]:.3f}, {best_params[3]:.3f}])")
    print(f"self.R = np.array([[{best_params[4]:.4f}]])")
