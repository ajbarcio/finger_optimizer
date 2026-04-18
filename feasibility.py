import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint, OptimizeResult
from scipy import optimize

def find_feasible_initial_guess(
    constraints_objects,
    bounds,
    x0_guess=None,
    num_attempts=10,
    method='least_squares'
):
    """
    Find a feasible initial guess that satisfies all constraints.
    
    Parameters:
    -----------
    constraints_objects : list
        List of LinearConstraint and NonlinearConstraint objects
    bounds : list of tuples
        Bounds for each variable [(lb1, ub1), (lb2, ub2), ...]
    x0_guess : array-like, optional
        Initial guess to start from (if None, uses midpoint of bounds)
    num_attempts : int
        Number of random starting points to try
    method : str
        'least_squares' or 'penalty'
    
    Returns:
    --------
    x_feasible : array
        Feasible point, or best attempt if no feasible point found
    is_feasible : bool
        Whether the point is truly feasible
    violations : dict
        Constraint violation details
    """
    
    num_vars = len(bounds)
    
    # Default initial guess: midpoint of bounds
    if x0_guess is None:
        x0_guess = np.array([(lb + ub) / 2 for lb, ub in bounds])
    
    def evaluate_constraint_violations(x):
        """Return total constraint violation"""
        violations = []
        
        for i, constraint in enumerate(constraints_objects):
            if isinstance(constraint, LinearConstraint):
                vals = constraint.A @ x
                lb = constraint.lb if hasattr(constraint.lb, '__len__') else np.full(vals.shape, constraint.lb)
                ub = constraint.ub if hasattr(constraint.ub, '__len__') else np.full(vals.shape, constraint.ub)
                
                # Violation is how much we're outside bounds
                lower_violation = np.maximum(0, lb - vals)
                upper_violation = np.maximum(0, vals - ub)
                violations.extend(lower_violation)
                violations.extend(upper_violation)
                
            elif isinstance(constraint, NonlinearConstraint):
                try:
                    val = constraint.fun(x)
                    if np.isscalar(val):
                        val = np.array([val])
                    
                    lb = constraint.lb if hasattr(constraint.lb, '__len__') else np.array([constraint.lb])
                    ub = constraint.ub if hasattr(constraint.ub, '__len__') else np.array([constraint.ub])
                    
                    lower_violation = np.maximum(0, lb - val)
                    upper_violation = np.maximum(0, val - ub)
                    violations.extend(lower_violation)
                    violations.extend(upper_violation)
                except Exception as e:
                    print(f"Warning: Error evaluating nonlinear constraint: {e}")
                    violations.append(1e6)  # Large penalty for evaluation failure
        
        return np.array(violations)
    
    def objective_least_squares(x):
        """Minimize sum of squared constraint violations"""
        violations = evaluate_constraint_violations(x)
        return np.sum(violations**2)
    
    def objective_penalty(x):
        """Minimize max violation (L-infinity norm)"""
        violations = evaluate_constraint_violations(x)
        return np.max(violations)
    
    # Try multiple random starting points
    best_x = None
    best_violation = np.inf
    
    for attempt in range(num_attempts):
        if attempt == 0:
            x_start = x0_guess
        else:
            # Random starting point within bounds
            x_start = np.array([
                np.random.uniform(lb, ub) 
                for lb, ub in bounds
            ])
        
        try:
            if method == 'least_squares':
                result = optimize.minimize(
                    objective_least_squares,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500}
                )
            else:  # penalty method
                result = optimize.minimize(
                    objective_penalty,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500}
                )
            
            total_violation = np.sum(evaluate_constraint_violations(result.x))
            
            if total_violation < best_violation:
                best_x = result.x
                best_violation = total_violation
            
            # If we found a feasible point, return immediately
            if total_violation < 1e-6:
                print(f"✓ Found feasible point on attempt {attempt + 1}")
                break
            else:
                print(f"I have tried to find a feasible point {attempt + 1} times")
        except Exception as e:
            print(f"Warning: Attempt {attempt + 1} failed: {e}")
            continue
    
    # Check feasibility and return detailed info
    is_feasible = best_violation < 1e-6
    
    violations_detail = {}
    for i, constraint in enumerate(constraints_objects):
        if isinstance(constraint, LinearConstraint):
            vals = constraint.A @ best_x
            lb = constraint.lb if hasattr(constraint.lb, '__len__') else np.full(vals.shape, constraint.lb)
            ub = constraint.ub if hasattr(constraint.ub, '__len__') else np.full(vals.shape, constraint.ub)
            
            violations_detail[f'linear_{i}'] = {
                'values': vals,
                'bounds': (lb, ub),
                'satisfied': np.all((vals >= lb - 1e-6) & (vals <= ub + 1e-6))
            }
            
        elif isinstance(constraint, NonlinearConstraint):
            try:
                val = constraint.fun(best_x)
                violations_detail[f'nonlinear_{i}'] = {
                    'value': val,
                    'bounds': (constraint.lb, constraint.ub),
                    'satisfied': (val >= constraint.lb - 1e-6) and (val <= constraint.ub + 1e-6)
                }
            except Exception as e:
                violations_detail[f'nonlinear_{i}'] = {
                    'error': str(e),
                    'satisfied': False
                }
    
    return best_x, is_feasible, violations_detail