import numpy as np
from desdeo_problem.problem.Constraint import ScalarConstraint
from desdeo_tools.solver import ScalarMethod
from scipy.optimize import minimize
from desdeo_problem.problem.Objective import  _ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Variable import variable_builder
from scipy.spatial.qhull import ConvexHull

from modules.GeometryDesign.tent import Tent
from modules import utils

def make_floor(p):
    p[-1] = [0,0,0]
    p[-2] = [1,0,0]
    p[-3] = [0,1,0]
    p[-4] = [1,1,0]
    return p

# Defining the objective functions
#Minimize
def surface_area(point_cloud_1d: np.ndarray, constant_floor=False) -> np.ndarray:
    point_cloud_1d = np.atleast_2d(point_cloud_1d)
    if constant_floor: point_cloud_1d[0,-12:] = [0,0,0, 1,0,0, 0,1,0, 1,1,0]
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud = utils.point_cloud_1d_to_3d(point_cloud_copy[0])
    t = Tent(point_cloud)
    return t.surface_area

#Maximize
def volume(point_cloud_1d: np.ndarray, constant_floor=False) -> np.ndarray:
    point_cloud_1d = np.atleast_2d(point_cloud_1d)
    if constant_floor: point_cloud_1d[0,-12:] = [0,0,0, 1,0,0, 0,1,0, 1,1,0]
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud = utils.point_cloud_1d_to_3d(point_cloud_copy[0])
    t = Tent(point_cloud)
    return t.volume

#Maximize
def min_height(point_cloud_1d: np.ndarray, constant_floor=False) -> np.ndarray:
    """
    Project point from floor along the z axis to the hull
    distance from the starting point to the projection point
    find the smallest
    """
    point_cloud_1d = np.atleast_2d(point_cloud_1d)
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud = utils.point_cloud_1d_to_3d(point_cloud_copy[0])
    t = Tent(point_cloud)
    return t.min_height

#Maximize
def floor_area(point_cloud_1d: np.ndarray, constant_floor=False) -> np.ndarray:
    point_cloud_1d = np.atleast_2d(point_cloud_1d)
    point_cloud_copy = np.copy(point_cloud_1d)
    point_cloud = utils.point_cloud_1d_to_3d(point_cloud_copy[0])
    t = Tent(point_cloud)
    return t.floor_area


def create_problem(var_count = 12, obj_mask = [True]*4, constraints = None, pfront = False, constant_floor = False):
    if constraints is not None:
        if constraints.shape[0] != 4 or constraints.shape[1] != 2:
            raise("invalid constraints")
        elif pfront: # Flip the values if pfront
            for i in range(1,4):
                np.flip(constraints[i])
                if constraints[i][0] is not None: constraints[i][0] = -constraints[i][0]
                if constraints[i][1] is not None: constraints[i][1] = -constraints[i][1]
    else:
        constraints = np.array([None] * 8).reshape((4,2))
    

    
    if type(obj_mask) is not np.ndarray:
        obj_mask = np.array(obj_mask)
        
    # objective functions
    surf = lambda xs: surface_area(xs, constant_floor)
    vol = lambda xs: volume(xs, constant_floor)
    height = lambda xs: min_height(xs, constant_floor)
    floor = lambda xs: floor_area(xs, constant_floor)

    # Objectives for desdeo problem
    obj1 = _ScalarObjective("surface_area", surf, maximize=False)
    obj2 = _ScalarObjective("volume", vol, maximize=True)
    obj3 = _ScalarObjective("min_height", height, maximize=True)
    obj4 = _ScalarObjective("floor_area", floor, maximize=True)

    # Objectives for pareto front solver variation. Minimizing all
    obj1_pfront = _ScalarObjective("surface_area", surf, maximize=False)
    obj2_pfront = _ScalarObjective("volume", lambda xs: -1*vol(xs), maximize=False)
    obj3_pfront = _ScalarObjective("min_height", lambda xs: -1*height(xs), maximize=False)
    obj4_pfront = _ScalarObjective("floor_area", lambda xs: -1*floor(xs), maximize=False)

    # List of objectives for MOProblem class
    objectives = np.array([obj1, obj2, obj3, obj4])[obj_mask]
    objectives_pfront = np.array([obj1_pfront, obj2_pfront, obj3_pfront, obj4_pfront,])[obj_mask]
    objectives_count = len(objectives)


    # Defining (decision) variables
    if constant_floor:
        var_count = var_count + 4

    actual_var_count = var_count * 3
    scale_factor = 1
    initial_values = scale_factor * (0.1 + (0.8 * np.random.rand(var_count, 3))) # random points

    initial_values = np.concatenate(initial_values)
    var_names = [f"point {i}.{axis}" for i in range(var_count) for axis in ['x', 'y', 'z']] 
    # var_names = [f"z{i}" for i in range(var_count)]



    # set lower bounds for each variable
    eps = 1e-04
    lower_bounds = scale_factor * np.array([0 - eps] * actual_var_count)

    # set upper bounds for each variable
    upper_bounds = scale_factor * np.array([1+ eps]  * actual_var_count)

    # Create a list of Variables for MOProblem class
    variables = variable_builder(var_names, initial_values, lower_bounds, upper_bounds)

    obj_f = [surface_area, volume, min_height, floor_area]
    cons = []
    for i in range(4):
        lower, upper = constraints[i]
        if lower is not None:
            con = utils.constraint_builder(obj_f[i], objectives_count, actual_var_count, lower, True, f"c{i}l")
            cons.append(con)
        if upper is not None:
            con = utils.constraint_builder(obj_f[i], objectives_count, actual_var_count, upper, False, f"c{i}u")
            cons.append(con)


    # Create the problem
    # This problem object can be passed to various different methods defined in DESDEO
    problem = MOProblem(objectives=objectives, variables=variables, constraints=cons)
    problem_pfront = MOProblem(objectives = objectives_pfront, variables=variables, constraints=cons)

    scipy_de_method = ScalarMethod(
        lambda x, _, **y: minimize(x, **y, x0 = np.random.rand(problem.n_of_variables)),
        method_args={"method":"SLSQP"},
        use_scipy=True
    )

    # Calculate ideal and nadir points
    #ideal, nadir = payoff_table_method(problem, solver_method=scipy_de_method)
    ideal = np.array([0, 1, 1, 1])[obj_mask]
    nadir = np.array([5, 0, 0, 0])[obj_mask]

    # Pass ideal and nadir to the problem object
    problem.ideal = ideal
    problem.nadir = nadir

    problem_pfront.ideal = -1*ideal
    problem_pfront.nadir = nadir

    return (problem_pfront if pfront else problem), scipy_de_method

def create_problem_constant_floor(var_count = 12, constraints = None, pfront = False):
    objective_mask = [True, True, False, False]
    if constraints is not None:
        constraints = np.vstack((constraints, np.array([[None] * 2]).reshape((1,2))))
    return create_problem(var_count, objective_mask, constraints, pfront, True)
