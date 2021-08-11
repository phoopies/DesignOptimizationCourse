from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from modules.utils import save
from modules.TwoBarTruss.problem import create_problem
import numpy as np
import warnings
warnings.filterwarnings("ignore") # ignore warnings :)

# Creating a two bar truss problem

# a constant load value for the problem
load = 65 

# Which objectives do you wish to optimize
# weight, stress, buckling stress and deflection
obj = np.array([
    True, True, True, True, # Optimizing all
])

# Approximate ideal and nadir
# ideal = 50, 3.59, 50, 0.015
# nadir = 580, 3.42, 100, 0.067


# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [100, None], # weight > 100
    [None, 15], # stress < 15
    [60, 100], # 60 < buckling < 100
    [None, None], # deflection no constraint
])

# To create the problem we can call the create_problem method with the parameters defined earlier
# The method returns a MOProblem and a scalarmethod instance which can be passed to different Desdeo objects
problem, method = create_problem(load, obj, constraints)


# Example on solving the pareto front : This might take some time so feel free to comment this out.

# We will use the solve_pareto_front_representation method but one can change this to something else.
# The method takes the problem instance and a step size array

# The method will create reference points from nadir to ideal with these step sizes
# large step sizes => less solutions but faster calculation
step_sizes = np.array([10, 20, 15, 2])[obj]

# The method returns the decision vectors and corresponding objective vectors
var, obj = solve_pareto_front_representation(problem, step_sizes, solver_method= method)

# save the solution if you wish, make sure to change the name to not accidentally overwrite an existing solution.
# Saved solutions can be used later to visualize it
# The solution will be saved to modules/DataAndVisualization/'name'
save("tbExample1", obj, var, problem.nadir, problem.ideal)
