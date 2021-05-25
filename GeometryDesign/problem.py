from desdeo_problem.Constraint import ScalarConstraint
from desdeo_problem.Problem import MOProblem
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Variable import variable_builder
from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
import numpy as np

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

# Defining the objective functions
#Minimize
def surface_area(point_cloud_1d: np.ndarray) -> np.ndarray:
    hull = form_hull_1d(point_cloud_1d[0])
    return hull.area # - floor_area(point_cloud)

#Maximize
def volume(point_cloud_1d: np.ndarray) -> np.ndarray:
    hull = form_hull_1d(point_cloud_1d[0])
    return hull.volume

#Maximize
def min_height(point_cloud_1d: np.ndarray) -> np.ndarray:
    """
    Project point from floor along the z axis to the hull
    distance from the starting point to the projection point
    find the smallest
    """
    return 0

#Maximize
def floor_area(point_cloud_1d: np.ndarray) -> np.ndarray:
    # How to define floor area? Project to z axis?
    hull = form_hull_1d(point_cloud_1d[0])
    return 0

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
    # Plotting the point cloud
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x,y,z = np.split(point_cloud, 3, 1)
    ax.scatter3D(x,y,z)

    #Plotting the hull
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(point_cloud[s, 0], point_cloud[s, 1], point_cloud[s, 2], "r-")
    plt.show()


# Defining (decision) variables
var_count = 100
var_names = [f"point {i}.{axis}" for i in range(var_count) for axis in ['x', 'y', 'z']] 

initial_values = np.random.rand(var_count, 3) # random points
initial_values = np.concatenate(initial_values)

# set lower bounds for each variable
lower_bounds = np.array([0] * var_count * 3)

# set upper bounds for each variable
upper_bounds = np.array([1] * var_count * 3)

# Create a list of Variables for MOProblem class
variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

# Define objectives
obj1 = _ScalarObjective("surface_area", surface_area, maximize=False)
obj2 = _ScalarObjective("volume", volume, maximize=True)
obj3 = _ScalarObjective("min_height", min_height, maximize=True)
obj4 = _ScalarObjective("floor_area", floor_area, maximize=True)

# List of objectives for MOProblem class
objectives = [obj1, obj2]
objectives_count = len(objectives)

# Define maximun values for objective functions
surface_area_max = 0
volume_max = 0
b_volume_max = 0
floor_area_max = 0

# Define constraint functions
"""
From desdeo_problem documentation:
The constraint should be defined so, that when evaluated, it should return a positive value, 
if the constraint is adhered to, and a negative, if the constraint is breached.
"""
const_surface_area = lambda xs, ys: surface_area_max - surface_area(xs)
const_volume = lambda xs, ys: volume_max - volume(xs)
const_min_height = lambda xs, ys: b_volume_max - min_height(xs)
const_floor_area = lambda xs, ys: floor_area_max - floor_area(xs)

# Define DESDEO ScalarConstraints
con1 = ScalarConstraint("surface_area", var_count, objectives_count, const_surface_area)
con2 = ScalarConstraint("volume", var_count, objectives_count, const_volume)
con3 = ScalarConstraint(
    "min_height", var_count, objectives_count, const_min_height
)
con4 = ScalarConstraint("floor_area", var_count, objectives_count, const_floor_area)

constraints = [con3, con4]  # List of constraints for MOProblem class

from desdeo_emo.EAs.RVEA import RVEA

# Create the problem
# This problem object can be passed to various different methods defined in DESDEO
problem = MOProblem(objectives=objectives, variables=variables)
print(problem.evaluate(initial_values))

# RVEA
from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_

evolver = RVEA(problem, interact=True, n_iterations=15, n_gen_per_iter=100)
figure = animate_init_(evolver.population.objectives, filename="geometry.html")
plot, pref = evolver.requests()

print(plot.content["dimensions_data"])

while evolver.continue_evolution():
    print(f"Current iteration {evolver._iteration_counter}")
    plot, pref = evolver.iterate()
    figure = animate_next_(
        plot.content['data'].values,
        figure,
        filename="geometry.html",
        generation=evolver._iteration_counter,
    )

print(plot.content['data'])


individuals = evolver.population.individuals
print(len(individuals))

for individual in individuals:
    point_cloud = point_cloud_1d_to_3d(individual)
    hull = ConvexHull(point_cloud)
    plot_hull(point_cloud ,hull)