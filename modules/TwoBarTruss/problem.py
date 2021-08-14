"""
Solving the Two-bar truss optimization problem with
DESDEO framework as defined in 
http://apmonitor.com/me575/uploads/Main/twobar.pdf
But in a multiobjective fashion:
Instead of just minimizing one objective,
try to minimize weight, stress, buckling stress and deflection simultaneously 


Take the time to convert everything to metric?
Currently imperial
H height (in)
d diameter (in)
t thickness (in)
B seperation distance (in)
E modulus of elasticity (1000lbs/in^2)
p density (lbs/in^3)
P load (1000 lbs)
"""

from modules import utils
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Objective import _ScalarObjective
from desdeo_problem.problem.Variable import variable_builder
from desdeo_mcdm.utilities.solvers import payoff_table_method
from scipy.optimize import  minimize
from desdeo_tools.solver import ScalarMethod
import numpy as np



def create_problem(load = 65, obj_mask = [True]*4, constraints = None):
    if constraints is not None:
        if constraints.shape[0] != 4 or constraints.shape[1] != 2:
            raise("invalid constraints")
    else:
        constraints = np.array([None] * 8).reshape((4,2))
    
    if type(obj_mask) is not np.ndarray:
        obj_mask = np.array(obj_mask)
    
    # Defining the objective functions
    def weight(xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(xs)
        H, d, t, B, E, p = xs.T  # Assign the values to named variables for clarity
        return p * 2 * np.pi * d * t * np.sqrt(np.square(B / 2) + np.square(H))


    def stress(xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(xs)
        H, d, t, B, E, p = xs.T
        numerator = load * np.sqrt(np.square(B / 2) + np.square(H))
        denominator = 2 * t * np.pi * d * H
        return numerator / denominator


    def buckling_stress(xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(xs)
        H, d, t, B, E, p = xs.T
        numerator = np.square(np.pi) * E * (np.square(d) + np.square(t))
        denominator = 8 * (np.square(B / 2) + np.square(H))
        return numerator / denominator


    def deflection(xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(xs)
        H, d, t, B, E, p = xs.T
        numerator = load * np.power(np.square(B / 2) + np.square(H), (3/2))
        denominator = 2 * t * np.pi * d * np.square(H) * E
        return numerator / denominator

    # Defining (decision) variables
    # Height, diameter, thickness, Seperation distance, modulus of elasticity, density, load
    var_names = ["H", "d", "t", "B", "E", "p"] 
    var_count = len(var_names)

    initial_values = np.array([30.0, 3.0, 0.1, 60.0, 30000., 0.3])

    # set lower bounds for each variable
    lower_bounds = np.array([20.0, 0.5, 0.01, 20.0, 25000., 0.01])

    # set upper bounds for each variable
    upper_bounds = np.array([60.0, 5.0, 1.0, 100.0, 40000., 0.5])

    # Trying to minimize everything so no need to define minimize array

    # Create a list of Variables for MOProblem class
    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    # Define objectives
    obj1 = _ScalarObjective("Weight", weight)
    obj2 = _ScalarObjective("Stress", stress)
    obj3 = _ScalarObjective("Buckling stress", buckling_stress)
    obj4 = _ScalarObjective("Deflection", deflection)

    # In this example we are minimizing all the objectives subject to buckling stress < 250 and deflection < 1

    # List of objectives for MOProblem class
    objectives = np.array([obj1, obj2, obj3, obj4])[obj_mask]
    objectives_count = len(objectives)

    # Define constraint functions

    obj_f = [weight, stress, buckling_stress, deflection]
    cons = []
    for i in range(4):
        lower, upper = constraints[i]
        if lower is not None:
            con = utils.constraint_builder(obj_f[i], objectives_count, var_count, lower, True, f"c{i}l")
            cons.append(con)
        if upper is not None:
            con = utils.constraint_builder(obj_f[i], objectives_count, var_count, upper, False, f"c{i}u")
            cons.append(con)


    # Create the problem
    # This problem object can be passed to various different methods defined in DESDEO
    prob = MOProblem(objectives=objectives, variables=variables, constraints=cons)

    scipy_de_method = ScalarMethod(
        lambda x, _, **y: minimize(x, **y, x0 = initial_values),
        method_args={"method":"SLSQP"},
        use_scipy=True
    )

    print("calculating ideal and nadir points")
    # Calculate ideal and nadir points
    ideal, nadir = payoff_table_method(prob, initial_guess=initial_values, solver_method=scipy_de_method)
    print(f"Nadir: {nadir}\nIdeal: {ideal}")

    # Pass ideal and nadir to the problem object
    prob.ideal = ideal
    prob.nadir = nadir

    return prob, scipy_de_method
