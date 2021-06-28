from desdeo_problem.Constraint import ScalarConstraint
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Objective import VectorObjective, _ScalarObjective
from desdeo_problem.Variable import variable_builder
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_mcdm.interactive.NIMBUS import NIMBUS
from desdeo_tools.scalarization.ASF import PointMethodASF
import numpy as np
from desdeo_mcdm.interactive.ReferencePointMethod import ReferencePointMethod
from desdeo_mcdm.interactive.NautilusNavigator import NautilusNavigator
from desdeo_mcdm.utilities import solve_pareto_front_representation
from desdeo_mcdm.utilities.solvers import payoff_table_method
from numpy.core.fromnumeric import nonzero, var
from scipy.optimize import differential_evolution, minimize
from desdeo_tools.solver import ScalarMethod
from tent import Tent
from desdeo_emo.EAs.RVEA import RVEA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from desdeo_tools.solver import ScalarMinimizer
import matplotlib.pyplot as plt
from desdeo_tools.scalarization import SimpleASF, Scalarizer
import pandas as pd
import pandas.plotting as pdplt

# Util functions
def form_floor_hull(point_cloud: np.ndarray):
    lowest_z = np.min(point_cloud[:,2])
    close_to_lowest_z = lambda z: np.abs(lowest_z - z) < 0.1 # Some small number
    floor_point_cloud = np.array([[x,y] for x,y,z in point_cloud if close_to_lowest_z(z)]) # 2D points of points which are close to floor level
    if len(floor_point_cloud) <= 2: # TODO check that forms an area, i.e all points not in same line
        return None
    return ConvexHull(floor_point_cloud)

def point_cloud_1d_to_3d(point_cloud_1d: np.ndarray):
    points_n = int(len(point_cloud_1d)/3)
    point_cloud_3d = point_cloud_1d.reshape(points_n, 3)
    return point_cloud_3d

def form_hull_1d(point_cloud_1d: np.ndarray):
    """
    Form a 3d convex hull from a point cloud
    where the point cloud is 1 dimensional array
    such that points at indices k, k+1, k+2 form 
    a point for all k = 0, 3, 9 ... , len(points)*3 
    """
    point_cloud_3d = point_cloud_1d_to_3d(point_cloud_1d)
    return ConvexHull(point_cloud_3d)

def plot_hull(point_cloud: np.ndarray, hull: ConvexHull):
    # Plotting the point cloud and the convex hull
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x,y,z = np.split(point_cloud, 3, 1)
    ax.scatter3D(x,y,z)

    #Plotting the hull
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(point_cloud[s, 0], point_cloud[s, 1], point_cloud[s, 2], "r-")
    plt.show()



# Defining the objective functions
#Minimize
def surface_area(point_cloud_1d: np.ndarray) -> np.ndarray:
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud_copy = np.atleast_2d(point_cloud_copy)
    point_cloud = point_cloud_1d_to_3d(point_cloud_copy[0])
    tent = Tent(point_cloud)
    # hull = form_hull_1d(point_cloud_1d[0])
    return tent.surface_area

#Maximize
def volume(point_cloud_1d: np.ndarray) -> np.ndarray:
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud_copy = np.atleast_2d(point_cloud_copy)
    point_cloud = point_cloud_1d_to_3d(point_cloud_copy[0])
    tent = Tent(point_cloud)
    # hull = form_hull_1d(point_cloud_1d[0])
    return tent.volume

#Maximize
def min_height(point_cloud_1d: np.ndarray) -> np.ndarray:
    """
    Project point from floor along the z axis to the hull
    distance from the starting point to the projection point
    find the smallest
    """
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud_copy = np.atleast_2d(point_cloud_copy)
    point_cloud = point_cloud_1d_to_3d(point_cloud_copy[0])
    tent = Tent(point_cloud)
    point_cloud_z = point_cloud[:,2]
    return np.min(point_cloud_z[np.nonzero(point_cloud_z)])

#Maximize
def floor_area(point_cloud_1d: np.ndarray) -> np.ndarray:
    point_cloud_copy = np.copy(point_cloud_1d)
    # How to define floor area? Project to z axis?
    point_cloud_copy = np.atleast_2d(point_cloud_copy)
    point_cloud = point_cloud_1d_to_3d(point_cloud_copy[0])
    # point_cloud_xy = np.column_stack((point_cloud[:,0], point_cloud[:,1]))
    # When input points are 2-dimensional, this is the area of the convex hull.
    tent = Tent(point_cloud)
    return tent.floor_area

# Define objectives
obj1 = _ScalarObjective("surface_area", surface_area, maximize=False)
obj2 = _ScalarObjective("volume", volume, maximize=True)
obj3 = _ScalarObjective("min_height", min_height, maximize=True)
obj4 = _ScalarObjective("floor_area", floor_area, maximize=True)

# List of objectives for MOProblem class
objectives = [obj1, obj2, obj4]
objectives_count = len(objectives)


# Defining (decision) variables
var_count = 15 # 3d points
actual_var_count = var_count * 3
scale_factor = 1
initial_values = scale_factor * (0.2 + (0.6 * np.random.rand(var_count, 3))) # random points

initial_values = np.concatenate(initial_values)
var_names = [f"point {i}.{axis}" for i in range(var_count) for axis in ['x', 'y', 'z']] 
# var_names = [f"z{i}" for i in range(var_count)]

# set lower bounds for each variable
eps = 1.e-1 # make sure that decision variable values don't exceed bounds because floats
lower_bounds = scale_factor * np.array([0-eps] * actual_var_count)
# lower_bounds = scale_factor * np.array([[0, 0, 0] for _ in range(var_count)])

# set upper bounds for each variable
upper_bounds = scale_factor * np.array([1+eps] * actual_var_count)

# Create a list of Variables for MOProblem class
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

# Define maximun values for objective functions
surface_area_max = 1900
volume_min = 0
min_height_min = 20
floor_area_min = 0.85 # Floor are should be bigger than some constant

# Define constraint functions
"""
From desdeo_problem documentation:
The constraint should be defined so, that when evaluated, it should return a positive value, 
if the constraint is adhered to, and a negative, if the constraint is breached.
"""
const_surface_area = lambda xs, ys: surface_area_max - surface_area(xs) # surface_area >= 0
const_volume = lambda xs, ys:  volume(xs) - volume_min
const_min_height = lambda xs, ys: min_height(xs) - min_height_min
const_floor_area = lambda xs, ys: -1. * floor_area_min + floor_area(xs) 

# Define DESDEO ScalarConstraints
con1 = ScalarConstraint("surface_area", actual_var_count, objectives_count, const_surface_area)
con2 = ScalarConstraint("volume", actual_var_count, objectives_count, const_volume)
con3 = ScalarConstraint(
    "min_height", actual_var_count, objectives_count, const_min_height
)
con4 = ScalarConstraint("floor_area", actual_var_count, objectives_count, const_floor_area)

constraints = [con4]  # List of constraints for MOProblem class


# Create the problem
# This problem object can be passed to various different methods defined in DESDEO
problem = MOProblem(objectives=objectives, variables=variables, constraints=constraints)

# Calculate ideal and nadir points
#ideal, nadir = payoff_table_method(problem, solver_method=scipy_de_method)
ideal = np.array([0., 1, 1])
nadir = np.array([5, 0, 0])
print(f"Nadir: {nadir}\nIdeal: {ideal}")

# Pass ideal and nadir to the problem object
problem.ideal = ideal
problem.nadir = nadir

scipy_de_method = ScalarMethod(
    lambda x, _, **y: minimize(x, **y, x0=initial_values), method_args={"method":"SLSQP"}, use_scipy=True
)

print(problem.evaluate(initial_values))

# Also dominated, should fix
def get_po_solutions(problem, step = 0.3, method = None):
    asf = PointMethodASF(problem.nadir, problem.ideal)
    scalarizer = Scalarizer(
        lambda x: problem.evaluate(x).objectives,
        asf,
        scalarizer_args={"reference_point": None},
    )

    solver = ScalarMinimizer(scalarizer, bounds= problem.get_variable_bounds() , method=method)

    stacked = np.stack((problem.ideal, problem.nadir)).T
    lower_slice_b, upper_slice_b = np.min(stacked, axis=1), np.max(stacked, axis=1)

    slices = [slice(start, stop + eps, step) for (start, stop) in zip(lower_slice_b, upper_slice_b)]

    z_mesh = np.mgrid[slices].reshape(len(problem.ideal), -1).T

    p_front_objectives = np.zeros(z_mesh.shape)
    p_front_variables = np.zeros((len(p_front_objectives), len(problem.get_variable_bounds().squeeze())))

    for i, z in enumerate(z_mesh):
        scalarizer._scalarizer_args = {"reference_point": z}
        res = solver.minimize(None)

        if not res["success"]:
            print("Non successfull optimization")
            p_front_objectives[i] = np.nan
            p_front_variables[i] = np.nan
            continue

        if np.any(res['x'] < 0):
            p_front_objectives[i] = np.nan
            p_front_variables[i] = np.nan
            continue

        f_i = problem.evaluate(res["x"]).objectives
        p_front_objectives[i] = f_i
        p_front_variables[i] = res["x"]

    return (
        p_front_variables[~np.all(np.isnan(p_front_variables), axis=1)],
        p_front_objectives[~np.all(np.isnan(p_front_objectives), axis=1)],
    )

savefile_name = "gd_surface_volume_floor__floor"
p_front_var, p_front_obj = solve_pareto_front_representation(problem, step = 0.05, solver_method = scipy_de_method)
np.savez(f"DataAndVisualization/{savefile_name}.npz", obj = p_front_obj, var = p_front_var)

# pof_variables, pof_objectives = solve_pareto_front_representation(problem, step = 0.1, solver_method=scipy_de_method2)
# print(f"Pareto front:\nVariables:\n{pof_variables}\nObjectives:\n{pof_objectives}")

# if len(pof_variables) > 0:
#     for vars in pof_variables:
#         t =  Tent(point_cloud_1d_to_3d(vars))
#         print (f"Area: {t.surface_area}\nVolume: {t.volume}\nFloorarea: {t.floor_area}")
#         t.plot()
# else:
#     print("No solutions found?")



# NSGAIII, Uncomment below to see solve with NSGAIII

# evolver = NSGAIII(problem, n_iterations=50, n_gen_per_iter=100, population_size=10)

# plot, pref = evolver.requests()

# while evolver.continue_evolution():
#     print(evolver._iteration_counter)
#     evolver.iterate()


# individuals = evolver.population.individuals

# individual = individuals[0]
# print(f"Some individual: {individual}")
# print(f"Objective values with above individual:\n"
#         f"Weight = {surface_area(individuals)}\n"
#         f"Stress = {volume(individuals[0])}\n"
#         f"Buckling stress = {min_height(individuals)}\n"
#         f"Deflection = {floor_area(individuals)}\n")


# RVEA
# from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_

# evolver = RVEA(problem, interact=False, n_iterations=25, n_gen_per_iter=100)
# figure = animate_init_(evolver.population.objectives, filename="geometry.html")
# plot, pref = evolver.requests()

# print(plot.content["dimensions_data"])

# while evolver.continue_evolution():
#     print(f"Current iteration {evolver._iteration_counter}")
#     plot, pref = evolver.iterate()
#     figure = animate_next_(
#         plot.content['data'].values,
#         figure,
#         filename="geometry.html",
#         generation=evolver._iteration_counter,
#     )

# print(plot.content['data'])

# individuals = evolver.population.individuals

# for individual in individuals:
#     point_cloud = point_cloud_1d_to_3d(individual)
#     t = Tent(point_cloud)
#     print (f"Area: {t.surface_area}\nVolume: {t.volume}\nFloorarea: {t.floor_area}")
#     t.plot()

# point_cloud = point_cloud_1d_to_3d(final_sol)
# t = Tent(point_cloud)
# print (f"Area: {t.surface_area}\nVolume: {t.volume}\nFloorarea: {t.floor_area}")
# t.plot()