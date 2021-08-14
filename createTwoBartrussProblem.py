from desdeo_mcdm.utilities.solvers import solve_pareto_front_representation
from modules.utils import save
from modules.TwoBarTruss.problem import create_problem
import numpy as np
import warnings
warnings.filterwarnings("ignore") # ignore warnings, the code is obviously perfect but just, you know... :)
# And still scipy float warnings arise...

# Creating a two bar truss problem

# a constant load value for the problem
load = 66 

# Which objectives do you wish to optimize
# weight, stress, buckling stress and deflection
obj = np.array([
    True, True, True, True, # Optimizing all
])

# Approximate ideal and nadir for a problem with no constraints
# nadir = 573, 2950, 535, 9
# ideal = 0.01, 2.1, 1.5, 0.01


# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Notice that breaking constraints will result in a penalty and therefore we might get results that break the constraints
constraints = np.array([
    [10, 100], #  10 < weight < 100
    [15, None], # stress > 15
    [None, 100], # buckling < 100
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
# The create_problem method below will print approximate values of the nadir and ideal
# This might help you set the step sizes to fit the problem.
step_sizes = np.array([100, 177, 100, 4])[obj]

# The method returns the decision vectors and corresponding objective vectors
var, obj = solve_pareto_front_representation(problem, step_sizes)

# save the solution if you wish, make sure to change the name to not accidentally overwrite an existing solution.
# Saved solutions can be used later to visualize it
# The solution will be saved to modules/DataAndVisualization/'name'
save("tbExample", obj, var, problem.nadir, problem.ideal)
