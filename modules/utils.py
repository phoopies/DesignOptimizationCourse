from typing import Union
from desdeo_problem.problem.Constraint import ScalarConstraint
from desdeo_problem.problem.Problem import MOProblem
import numpy as np
from numpy.core.records import array
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from desdeo_problem.problem.Objective import _ScalarObjective

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

def save(name, objectives, decision, nadir, ideal):
    np.savez(
        f"modules/DataAndVisualization/{name}.npz",
        obj = objectives,
        var = decision,
        nadir = nadir,
        ideal = ideal
    )
    print("Saved successfully")

def load(name):
    d = np.load(f"modules/DataAndVisualization/{name}.npz")
    if d['nadir'] is not None and d['ideal'] is not None:
        return d['obj'], d['var'], d['nadir'], d['ideal']
    else:
        return d['obj'], d['var'], None, None


# """
# From desdeo_problem documentation:
# The constraint should be defined so, that when evaluated, it should return a positive value, 
# if the constraint is adhered to, and a negative, if the constraint is breached.
# """
def constraint_builder(f, n_obj, n_var, bound, is_lower_bound = True, name= "c1"):
    c = lambda xs, ys: f(xs) - bound if is_lower_bound else bound - f(xs) 
    return ScalarConstraint(name, n_var, n_obj, c)