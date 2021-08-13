from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from modules.utils import save
from modules.GeometryDesign.problem import create_problem_constant_floor as create_problem
import numpy as np
import warnings
warnings.filterwarnings("ignore") # ignore warnings :)

# Creating geometry design problem with constant floor : tent like buildings

# Always optimizing surface area and volume

# ideal and nadir in respective order
# ideal = 0, 1
# nadir = 5, 0


# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [0.2, None], # Surface area > 0.2
    [.5, .8], # .5 < volume < .8. Even though we're not optimizing volume, we can set a constraint on it  
    [.4, None], # Min height > 0.4
])

# How many 3d points should the hull be formed of
# more points => More complex problem : longer execution times
# Less points => More likely to fail in constructing the hull
variable_count = 12 # Around 10 - 25 seems to be good enough

# To create the problem we can call the gd_create method with the parameters defined earlier
# the pfront argument should be set to True if using the solve_pareto_front_representation method as it doesn't 
# take account minimizing/maximizing. For everything else we can set it to False
# The method returns a MOProblem and a scalarmethod instance which can be passed to different Desdeo objects
problem, method = create_problem(variable_count, constraints, pfront = True)


# Example on solving the pareto front : This might take some time so feel free to comment this out (lines 54, 57 and 62).

# We will use the solve_pareto_front_representation method but one can change this to something else.
# The method takes the problem instance and a step size array

# The method will create reference points from nadir to ideal with these step sizes
# in this case : ref points = [[5,0], [4.5, 0], [4, 0] ... [5, 0.2], [4.8, 0.2] ... [0, 1]]
# large step sizes => less solutions but faster calculation
step_sizes = np.array([1, .5])

# The method returns the decision vectors and corresponding objective vectors
var, obj = solve_pareto_front_representation(problem, step_sizes, solver_method= method)

# save the solution if you wish, make sure to change the name to not accidentally overwrite an existing solution.
# Saved solutions can be used later to visualize it
# The solution will be saved to modules/DataAndVisualization/'name'
save("gdcExample1", obj, var, problem.nadir, problem.ideal)
