#!/usr/bin/env python3
"""
Batch Bayesian Optimization for PD Tuning
Suggests multiple parameter sets per iteration for parallel testing
"""

import numpy as np
import json
from skopt import Optimizer
from skopt.space import Real
import matplotlib.pyplot as plt


class BatchBayesianTuner:
    def __init__(self, batch_size=4):
        """
        Initialize batch Bayesian optimizer
        
        Args:
            batch_size: Number of parameter sets to suggest per iteration
        """
        self.batch_size = batch_size
        
        # Define search space
        self.space = [
            Real(0.1, 10.0, name='Kp_linear'),
            Real(0.01, 2.0, name='Kd_linear'),
            Real(0.1, 10.0, name='Kp_angular'),
            Real(0.01, 2.0, name='Kd_angular'),
        ]
        
        # Create optimizer
        self.optimizer = Optimizer(self.space, n_initial_points=10)
        print("Adding default baseline parameters to optimizer...")
        default_X = [1.0, 0.1, 1.0, 0.1]  # Kp_lin, Kd_lin, Kp_ang, Kd_ang
        default_avg_error = 0.3156
        default_max_error = 1.1353
        default_objective = 0.7 * default_avg_error + 0.3 * default_max_error
        self.optimizer.tell(default_X, default_objective)
        print(f"Baseline: avg_err={default_avg_error:.4f}, max_err={default_max_error:.4f}, obj={default_objective:.4f}")
        self.iteration = 0
        self.all_results = []
	
    def suggest_batch(self):
    	"""
    	Suggest a batch of parameter sets to test
    
    	Returns:
        	list: List of parameter dictionaries
    	"""
    	# Ask for all points at once to get diverse samples
    	params_arrays = self.optimizer.ask(n_points=self.batch_size)
    
    	batch = []
    	for i, params_array in enumerate(params_arrays):
        	params_dict = {
            	'turtle_id': i + 1,
            	'Kp_linear': params_array[0],
            	'Kd_linear': params_array[1],
            	'Kp_angular': params_array[2],
            	'Kd_angular': params_array[3],
        	}
        	batch.append(params_dict)
    
    	return batch    
    
    def record_batch_results(self, results):
        """
        Record results from a batch of tests
        
        Args:
            results: List of dicts with keys: params, avg_error, max_error
        """
        for result in results:
            params = result['params']
            avg_error = result['avg_error']
            max_error = result['max_error']
            
            # Convert params dict to array for optimizer
            X = [
                params['Kp_linear'],
                params['Kd_linear'],
                params['Kp_angular'],
                params['Kd_angular']
            ]
            
            # Calculate objective
            y = 0.7 * avg_error + 0.3 * max_error
            
            # Tell optimizer
            self.optimizer.tell(X, y)
            
            # Store result
            result_record = {
                'iteration': self.iteration,
                'turtle_id': params.get('turtle_id', 0),
                'params': params,
                'avg_error': avg_error,
                'max_error': max_error,
                'objective': y
            }
            self.all_results.append(result_record)
    
    def get_best_params(self):
        """Get the best parameters found so far"""
        if not self.all_results:
            return None
        
        best_idx = np.argmin([r['objective'] for r in self.all_results])
        return self.all_results[best_idx]
    
    def save_results(self, filename='batch_bayesian_results.json'):
        """Save all results to file"""
        with open(filename, 'w') as f:
            json.dump({
                'all_results': self.all_results,
                'best': self.get_best_params()
            }, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def plot_progress(self, filename='batch_optimization_progress.png'):
        """Plot optimization progress"""
        if len(self.all_results) < 2:
            print("Not enough data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data
        iterations = [r['iteration'] for r in self.all_results]
        objectives = [r['objective'] for r in self.all_results]
        avg_errors = [r['avg_error'] for r in self.all_results]
        max_errors = [r['max_error'] for r in self.all_results]
        
        # Plot 1: Objective over time
        axes[0, 0].scatter(iterations, objectives, alpha=0.6)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective Value')
        axes[0, 0].set_title('Objective Function')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Best so far
        best_so_far = []
        best_val = float('inf')
        for obj in objectives:
            if obj < best_val:
                best_val = obj
            best_so_far.append(best_val)
        
        axes[0, 1].plot(range(len(best_so_far)), best_so_far, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Test Number')
        axes[0, 1].set_ylabel('Best Objective')
        axes[0, 1].set_title('Best Objective So Far')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Average Error
        axes[0, 2].scatter(iterations, avg_errors, alpha=0.6, c=objectives, cmap='viridis')
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Avg Cross-Track Error')
        axes[0, 2].set_title('Average Error')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4-7: Parameter evolution
        param_names = ['Kp_linear', 'Kd_linear', 'Kp_angular', 'Kd_angular']
        for idx, (ax, param_name) in enumerate(zip(
            [axes[1, 0], axes[1, 1], axes[1, 2]],
            param_names[:3]
        )):
            values = [r['params'][param_name] for r in self.all_results]
            ax.scatter(iterations, values, alpha=0.6, c=objectives, cmap='viridis')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param_name)
            ax.set_title(f'{param_name} Evolution')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
        plt.close()


def interactive_batch_mode():
    """Run in interactive batch mode"""
    print("\n" + "="*60)
    print("BATCH BAYESIAN OPTIMIZATION FOR PD CONTROLLER TUNING")
    print("Multi-Turtle Mode")
    print("="*60)
    
    # Get batch size from user
    while True:
        try:
            batch_size = int(input("\nHow many turtles to test simultaneously? (2-6): ").strip())
            if 2 <= batch_size <= 6:
                break
            print("Please enter a number between 2 and 6")
        except ValueError:
            print("Please enter a valid number")
    
    tuner = BatchBayesianTuner(batch_size=batch_size)
    
    print(f"\n{'='*60}")
    print(f"Configuration: Testing {batch_size} parameter sets per iteration")
    print("="*60)
    print("\nWorkflow:")
    print("1. I'll suggest multiple parameter sets")
    print("2. You spawn turtles and assign parameters to each")
    print("3. Run all turtles simultaneously")
    print("4. Report back the results for each turtle")
    print("5. Repeat!")
    print("\nType 'done' when finished, 'best' to see best params")
    print("="*60 + "\n")
    
    while True:
        tuner.iteration += 1
        
        print("\n" + "="*60)
        print(f"ITERATION {tuner.iteration}")
        print("="*60)
        
        # Suggest batch of parameters
        batch = tuner.suggest_batch()
        
        print("\nTest these parameter sets:")
        print("-" * 60)
        for params in batch:
            print(f"\nTurtle {params['turtle_id']}:")
            print(f"  Kp_linear:  {params['Kp_linear']:.4f}")
            print(f"  Kd_linear:  {params['Kd_linear']:.4f}")
            print(f"  Kp_angular: {params['Kp_angular']:.4f}")
            print(f"  Kd_angular: {params['Kd_angular']:.4f}")
        
        print("\n" + "-" * 60)
        print("\nSpawn your turtles and run the tests!")
        print("Then enter results for each turtle below.")
        print("(Press Enter after each turtle's results)\n")
        
        # Collect results for each turtle in the batch
        batch_results = []
        
        for i, params in enumerate(batch):
            print(f"\nTurtle {params['turtle_id']} results:")
            
            try:
                input_str = input("  Enter avg_error,max_error (or 'done'/'best'): ").strip()
                
                if input_str.lower() == 'done':
                    # Finish optimization
                    break
                elif input_str.lower() == 'best':
                    # Show best so far
                    best = tuner.get_best_params()
                    if best:
                        print("\n" + "="*60)
                        print("BEST PARAMETERS SO FAR:")
                        print("="*60)
                        print(f"  Kp_linear:  {best['params']['Kp_linear']:.4f}")
                        print(f"  Kd_linear:  {best['params']['Kd_linear']:.4f}")
                        print(f"  Kp_angular: {best['params']['Kp_angular']:.4f}")
                        print(f"  Kd_angular: {best['params']['Kd_angular']:.4f}")
                        print(f"  Objective:  {best['objective']:.4f}")
                        print(f"  Avg Error:  {best['avg_error']:.4f}")
                        print(f"  Max Error:  {best['max_error']:.4f}")
                        print("="*60 + "\n")
                    # Ask again for this turtle
                    i -= 1
                    continue
                
                # Parse comma-separated values
                avg_error, max_error = map(float, input_str.split(','))
                
                batch_results.append({
                    'params': params,
                    'avg_error': avg_error,
                    'max_error': max_error
                })
                
                print(f"  ✓ Recorded: avg={avg_error:.4f}, max={max_error:.4f}")
                
            except ValueError:
                print("  ✗ Invalid input. Please enter: avg_error,max_error")
                i -= 1  # Retry this turtle
                continue
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
        
        if input_str.lower() == 'done' or not batch_results:
            break
        
        # Record the batch results
        tuner.record_batch_results(batch_results)
        
        # Show summary
        print("\n" + "-" * 60)
        print("BATCH SUMMARY:")
        for result in batch_results:
            objective = 0.7 * result['avg_error'] + 0.3 * result['max_error']
            print(f"  Turtle {result['params']['turtle_id']}: objective={objective:.4f}")
        
        best_in_batch = min(batch_results, key=lambda x: 0.7*x['avg_error'] + 0.3*x['max_error'])
        print(f"\n  Best in this batch: Turtle {best_in_batch['params']['turtle_id']}")
        print("-" * 60)
        
        # Save progress
        tuner.save_results()
    
    # Final results
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    
    best = tuner.get_best_params()
    if best:
        print("\nBest parameters found:")
        print(f"  Kp_linear:  {best['params']['Kp_linear']:.4f}")
        print(f"  Kd_linear:  {best['params']['Kd_linear']:.4f}")
        print(f"  Kp_angular: {best['params']['Kp_angular']:.4f}")
        print(f"  Kd_angular: {best['params']['Kd_angular']:.4f}")
        print(f"\nBest objective: {best['objective']:.4f}")
        print(f"Avg error: {best['avg_error']:.4f}")
        print(f"Max error: {best['max_error']:.4f}")
        print(f"\nFrom iteration {best['iteration']}, Turtle {best['turtle_id']}")
    
    # Save and plot
    tuner.save_results()
    tuner.plot_progress()
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    interactive_batch_mode()
