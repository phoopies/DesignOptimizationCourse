from  modules.DataAndVisualization.vizualiser import (
    plot_scatter as scatter,
    plot_parallel as parallel,
    visualize as interactive_scatter_gd,
)
from modules.utils import (
    save, load,
)
from modules.GeometryDesign.problem import (
    create_problem as gd_create,
    create_problem_constant_floor as gd_create_floor
)
from modules.TwoBarTruss.problem import create_problem as tb_create
from desdeo_mcdm.utilities import solve_pareto_front_representation
from desdeo_tools.solver import ScalarMethod
from scipy.optimize import minimize
import numpy as np

# The method used to solve the achievement scalarizing function
# You can leave this as is.
scipy_de_method = ScalarMethod(
    lambda x, _, **y: minimize(x, **y, x0 = np.random.rand(problem.n_of_variables)),
    method_args={"method":"SLSQP"},
    use_scipy=True
)

# You can create the problem you wish with desired values 
# And calculate the pareto front using the solve_pareto_front_representation function
# OR use some interactive method from DESDEO to solve the problem
# To do this import the method you wish to use and 
# pass the created problem to the method and continue with the steps in the method.


# Or you can use precalculated solutions, see modules/DataAndVisualization

# To plot solutions use scatter for a 2 or 3d scatter plot
# parallel for multidimensional solutions
# interactive scatter for 2 or 3d scatter with clickable points -> convex hull of corresponding decision variables


# Example on creating problems:

# Geometry design problem, tent like buildings
# Which objectives do you wish to optimize
# surface area, volume, min height and floor area
obj_gd = np.array([
    True, True, False, True,
])


# Set constraint for objectives, [lower, upper]
# If no constraint then set it to None
# Each row represents a objective function in the same order as in obj_gd 
# If solving the pareto front with solve_pareto_front_representation function
# then every objective is to be minimized -> Make sure to flip the boundaries around
# for objectives to be maximized. i.e volume (maximize) regular bounds [0,1]
# flip -> [-1, 0] (minimize)
constraints_gd = np.array([
    [0, 0.2],
    [0.5, 0.7],
    [None, None],
    [0.75, None]
])

# How many 3d points should the hull be formed of
# The more you have the longer it takes to calculate the Pareto front
# less than 5 might cause issues in constructing the hull.
variable_count_gd = 12

# Create the problem, 
problem = gd_create(variable_count_gd , obj_gd, constraints_gd)


# Solve the pareto front representation, this might take a long time 
# var, obj = solve_pareto_front_representation(problem, 0.5, solver_method= scipy_de_method)
# save the values if you wish, make sure to change the name "ex1" to something better
# save("ex1", obj, var, problem.nadir, problem.ideal)



# Geometry design with constant floor
# This is a 2 dimension problem, only optimizing surface area and volume
constraints_gd_floor = np.array([
    [None, None], 
    [None, None]
])

# Create a problem for calculating pareto front with 10 + 4 3d Points without constraints
problem_gd_floor = gd_create_floor(15, constraints_gd_floor, pfront=True)
step_sizes = np.array([0.2, 0.05])

var, obj = solve_pareto_front_representation(problem_gd_floor)
save("gd_constant_floor__None", obj, var, problem_gd_floor.nadir, problem_gd_floor.ideal)
interactive_scatter_gd(obj, var, ["Volume", "Surface_area"])

# two bar truss problem
# weight, stress, buckling stress and deflection
obj_tb = np.array([
    True, True, True, True,
])

# Constraints
constraints_tb = np.array([
    [None, None],
    [None, None],
    [None, None],
    [None, None],
])

# load = 65
# tb_problem = tb_create(load, obj_tb, constraints_tb, method = scipy_de_method)
# var, obj = solve_pareto_front_representation(tb_problem, 0.1, solver_method = scipy_de_method)
# save("ex2", obj, var, tb_problem.nadir, tb_problem.ideal)

# END OF EXAMPLE


# Example of plotting problems:

# Load solution or solve yourself
obj, var, nadir, ideal = load("gd_surface_volume_floor__none")

obj = abs(obj) # Make sure all values are positive as some of the objectives were flipped.
ideal = abs(ideal) # Same for ideal

# Axis names for the plot
axis_names_gd = ["Surface area", "Volume", "Floor area"]

# Axis ranges for the plot
axis_ranges_gd = np.stack((nadir, ideal), axis = 1)

# For gd problems with 2 or 3d you can use the interactive plot:
interactive_scatter_gd(obj, var, axis_names_gd, axis_ranges_gd)

# Load a 4d solution
obj, var, nadir, ideal = load("gd_tent__all__none")
# for >3 dimension use the parallel plot
# parallel(obj) # TODO axis names

# END OF EXAMPLE