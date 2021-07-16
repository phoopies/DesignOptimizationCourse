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
import numpy as np

# I AM NOT SURE IF THE CONSTRAINTS ARE CURRENTLY WORKING
# WILL BE LOOKING INTO THIS SOON
# ALSO I MODIFIED MIN HEIGHT FUNCTION AND I STILL NEED TO VERIFY IT WORKS AS IT SHOULD

# You can create the problem you wish with desired values THEN
# you either calculate the pareto front using the solve_pareto_front_representation function
# OR use some interactive method from DESDEO to solve the problem
# To do this import the method you wish to use and 
# pass the created problem to the method and continue with the steps in the method.


# Or you can use precalculated solutions, see modules/DataAndVisualization

# To plot solutions with less 2 or 3 dimensions you can use the scatter plot
# parallel for any dimension solutions

# interactive scatter for geometry design problems with 2 or 3dimension
# -> scatter plot with clickable points -> convex hull of corresponding decision variables


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
constraints_gd = np.array([
    [1, 5],
    [0.5, 1], 
    [None, None],
    [0.75, None],
])

# How many 3d points should the hull be formed of
# The more you have the longer it takes to calculate the Pareto front
# less than 5 might cause issues in constructing the hull.
variable_count_gd = 12

# Create the problem, save the solver method for solve_pareto.... function. 
# Remember to set pfront to True if wanting to solve the pfront
problem_gd, method_gd = gd_create(variable_count_gd , obj_gd, constraints_gd, pfront = True)



# Solve the pareto front representation, this might take a long time 
# var, obj = solve_pareto_front_representation(problem_gd, np.array([1, 0.5, 0.5]), solver_method= method_gd)
# save the values if you wish, make sure to change the name "ex1" to something better
# save("ex1", obj, var, problem_gd.nadir, problem_gd.ideal)
# interactive_scatter_gd(obj, var)


# Geometry design with constant/predefined floor with area of 1
# This is a 2 dimension problem, only optimizing surface area and volume

# Set the constraint for surface area and volume
constraints_gd_floor = np.array([
    [None, None], 
    [None, None]
])

# Create a problem for calculating pareto front with 10 + 4 3d Points without constraints
# The less points one has the more likely it is that constructing the convex hull fails
problem_gd_floor, method_gd_floor = gd_create_floor(10, constraints_gd_floor, pfront=True)
# step_sizes = np.array([.35, .1])

# var, obj = solve_pareto_front_representation(problem_gd_floor, step = step_sizes, solver_method = method_gd_floor)
# save("gd_constant_floor__None1", obj, var, problem_gd_floor.nadir, problem_gd_floor.ideal)

# interactive_scatter_gd(obj, var, ["Surface_area", "Volume"])



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

load_tb = 65
tb_problem, solver_method = tb_create(load_tb, obj_tb, constraints_tb)
var, obj = solve_pareto_front_representation(tb_problem, 5.0, solver_method = solver_method)
# save("ex2", obj, var, tb_problem.nadir, tb_problem.ideal)

# END OF EXAMPLE


# Example of plotting problems:

# Load solution or solve yourself
# We're loading a presolved solution. More info on these WILL be found on the DataAndVisualization folder.
obj, var, nadir, ideal = load("gd_surface_volume_floor__none")

# I guess this part could be before saving the values. Well this can be changed later on
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


# Load a constant floor gd problem and plot it with interactive plot
obj, var, nadir, ideal = load("gd_constant_floor__None1")
interactive_scatter_gd(obj, var, ["Surface", "Volume"])

# END OF EXAMPLE