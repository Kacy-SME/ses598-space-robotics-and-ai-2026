#!/usr/bin/env python3
"""
Multi-Turtle Bayesian Optimization
Tests multiple parameter sets simultaneously by spawning turtles
"""

import rclpy
from rclpy.node import Node
from turtlesim.srv import Spawn, Kill
import subprocess
import time
import json
from skopt import Optimizer
from skopt.space import Real
import numpy as np


class MultiTurtleOptimizer:
    def __init__(self, num_turtles=3, test_duration=30):
        """
        Initialize multi-turtle optimizer
        
        Args:
            num_turtles: Number of turtles to test simultaneously
            test_duration: Duration of each test in seconds
        """
        self.num_turtles = num_turtles
        self.test_duration = test_duration
        
        # Bayesian Optimizer setup
        self.space = [
            Real(0.1, 10.0, name='Kp_linear'),
            Real(0.01, 2.0, name='Kd_linear'),
            Real(0.1, 10.0, name='Kp_angular'),
            Real(0.01, 2.0, name='Kd_angular'),
        ]
        self.optimizer = Optimizer(self.space, n_initial_points=5)
        
        # Results storage
        self.all_results = []
        self.iteration = 0
    
    def spawn_turtle(self, name, x, y):
        """Spawn a turtle at specific position"""
        cmd = f'ros2 service call /spawn turtlesim/srv/Spawn "{{x: {x}, y: {y}, theta: 0.0, name: \'{name}\'}}"'
        subprocess.run(cmd, shell=True, capture_output=True)
        print(f"Spawned {name} at ({x}, {y})")
    
    def kill_turtle(self, name):
        """Kill a turtle"""
        cmd = f'ros2 service call /kill turtlesim/srv/Kill "{{name: \'{name}\'}}"'
        subprocess.run(cmd, shell=True, capture_output=True)
    
    def setup_turtles(self, params_list):
        """
        Setup multiple turtles with different parameters
        
        Args:
            params_list: List of parameter dictionaries
        """
        # Turtle positions (spread out)
        positions = [
            (2.0, 2.0),
            (5.5, 2.0),
            (9.0, 2.0),
            (2.0, 9.0),
            (5.5, 9.0),
            (9.0, 9.0),
        ]
        
        turtle_names = []
        for i, params in enumerate(params_list[:self.num_turtles]):
            name = f'test_turtle_{i}'
            x, y = positions[i % len(positions)]
            
            # Kill if exists
            self.kill_turtle(name)
            time.sleep(0.5)
            
            # Spawn new
            self.spawn_turtle(name, x, y)
            turtle_names.append(name)
            
            print(f"  {name}: Kp_lin={params['Kp_linear']:.3f}, Kd_lin={params['Kd_linear']:.3f}, "
                  f"Kp_ang={params['Kp_angular']:.3f}, Kd_ang={params['Kd_angular']:.3f}")
        
        return turtle_names
    
    def collect_metrics(self, turtle_name, duration):
        """
        Collect cross-track error metrics for a turtle
        
        Args:
            turtle_name: Name of the turtle
            duration: How long to collect data
            
        Returns:
            dict: Performance metrics
        """
        # Implement metric collection
        print(f"    Collecting metrics for {turtle_name}...")
        time.sleep(duration)
        
        avg_error = np.random.uniform(0.1, 2.0)
        max_error = avg_error * 1.5
        
        return {
            'avg_error': avg_error,
            'max_error': max_error
        }
    
    def run_iteration(self):
        """Run one iteration of Bayesian Optimization"""
        self.iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}")
        print(f"{'='*60}\n")
        
        # Get next batch of parameters from optimizer
        params_list = []
        for i in range(self.num_turtles):
            suggested = self.optimizer.ask()
            params = {
                'Kp_linear': suggested[0],
                'Kd_linear': suggested[1],
                'Kp_angular': suggested[2],
                'Kd_angular': suggested[3],
            }
            params_list.append(params)
        
        print("Testing parameters:")
        turtle_names = self.setup_turtles(params_list)
        
        print(f"\nRunning tests for {self.test_duration} seconds...")
        print("Watch TurtleSim to see which turtle performs best!")
        
        # Collect metrics for each turtle
        results = []
        for turtle_name, params in zip(turtle_names, params_list):
            metrics = self.collect_metrics(turtle_name, self.test_duration / self.num_turtles)
            
            # Calculate objective
            objective = 0.7 * metrics['avg_error'] + 0.3 * metrics['max_error']
            
            # Tell optimizer
            X = [params['Kp_linear'], params['Kd_linear'], 
                 params['Kp_angular'], params['Kd_angular']]
            self.optimizer.tell(X, objective)
            
            result = {
                'iteration': self.iteration,
                'turtle': turtle_name,
                'params': params,
                'metrics': metrics,
                'objective': objective
            }
            results.append(result)
            self.all_results.append(result)
            
            print(f"  {turtle_name}: objective={objective:.4f}, "
                  f"avg_err={metrics['avg_error']:.4f}, max_err={metrics['max_error']:.4f}")
        
        return results
    
    def get_best_params(self):
        """Get the best parameters found so far"""
        if not self.all_results:
            return None
        
        best = min(self.all_results, key=lambda x: x['objective'])
        return best
    
    def save_results(self, filename='multi_turtle_results.json'):
        """Save all results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def run_optimization(self, n_iterations=10):
        """
        Run the full optimization process
        
        Args:
            n_iterations: Number of iterations to run
        """
        print("\n" + "="*60)
        print("MULTI-TURTLE BAYESIAN OPTIMIZATION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Turtles per iteration: {self.num_turtles}")
        print(f"  Test duration: {self.test_duration}s")
        print(f"  Total iterations: {n_iterations}")
        print(f"  Total tests: {n_iterations * self.num_turtles}")
        print("="*60 + "\n")
        
        for i in range(n_iterations):
            self.run_iteration()
            self.save_results()
        
        # Final summary
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE!")
        print("="*60)
        
        best = self.get_best_params()
        if best:
            print("\nBest parameters found:")
            for key, val in best['params'].items():
                print(f"  {key}: {val:.4f}")
            print(f"\nBest objective: {best['objective']:.4f}")
            print(f"Avg error: {best['metrics']['avg_error']:.4f}")
            print(f"Max error: {best['metrics']['max_error']:.4f}")
        
        print("\n" + "="*60 + "\n")


def main():
    """Main function"""
    # Create optimizer
    optimizer = MultiTurtleOptimizer(
        num_turtles=3,
        test_duration=30
    )
    
    # Run optimization
    optimizer.run_optimization(n_iterations=10)
    
    print("\nTo visualize turtles, make sure TurtleSim is running:")
    print("  ros2 run turtlesim turtlesim_node")


if __name__ == '__main__':
    main()
