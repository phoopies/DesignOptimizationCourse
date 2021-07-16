from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial.qhull import QhullError


class Tent:
    _point_cloud: np.ndarray
    main_hull: ConvexHull
    floor_hull: ConvexHull

    def __init__(self, point_cloud) -> None:
        self._point_cloud = point_cloud
        self.make_hull()

    @property
    def floor_area(self):
        return self.floor_hull.volume

    @property
    def surface_area(self):
        return self.main_hull.area - self.floor_area

    @property
    def volume(self):
        return self.main_hull.volume
    
    @property
    def min_height(self):
        z = self._point_cloud[self.main_hull.simplices][:,2]
        z = z[z > 0]
        return np.min(z)
    
    def plot(self): #TODO Opaque, polygons
        ax = plt.axes(projection='3d')
        x,y,z = np.split(self._point_cloud, 3, 1)
        ax.scatter3D(x,y,z)
        ax.xlim = [0,1]

        #Plotting the hull
        corners = []
        for s in self.main_hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(self._point_cloud[s, 0], self._point_cloud[s, 1], self._point_cloud[s, 2], "r-")
            corners.append(s)

        for v in self.floor_hull.simplices:
            v = np.append(v, v[0])
            ax.plot(self._point_cloud[v, 0], self._point_cloud[v, 1])

        triangles = self._point_cloud[self.main_hull.simplices]
        patches = Poly3DCollection(triangles)
        patches.set_alpha(0.5)
        patches.set_facecolor("green")
        patches.set_edgecolor("black")
        ax.add_collection3d(patches)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        plt.show()

    def make_floor(self):
        point_cloud_2d = np.delete(self._point_cloud,2,1) # Delete z axis
        floor_hull = ConvexHull(point_cloud_2d) # Make the 2d hull
        floor_corners = np.unique(floor_hull.simplices)
        self._point_cloud[floor_corners, 2] = 0
        self.floor_hull = floor_hull
    
    def make_hull(self):
        self.make_floor()
        try:
            hull = ConvexHull(self._point_cloud)
        except QhullError:
            print("Failed to construct the hull, offsetting points")
            print("failed point cloud: ", self._point_cloud)
            self.offset_points()
            return self.make_hull() # Try again with offset points
        self.main_hull = hull
    
    # TODO make sure that doesn't go over bounds
    def offset_points(self): 
        eps = 1e-04
        sh = self._point_cloud.shape
        self._point_cloud = self._point_cloud + np.random.rand(sh) * eps 

# TODO REMOVE
# def floor_hull(point_cloud) -> ConvexHull:
#     """
#     """
#     point_cloud_2d = np.delete(point_cloud,2,1) # Delete z axis
#     floor_hull = ConvexHull(point_cloud_2d) # Make the 2d hull
#     return floor_hull

# # Better name!
# def make_tent(point_cloud):
#     """
#     Constructs a 2d convex hull from x,y points and projects them to floor lever z = 0
#     Then constructs a convex hull from new points (with the floor)

#     Returns:
#         ConvexHull of 3d point cloud and the floor "hull"
#     """
#     floor = floor_hull(point_cloud)
#     floor_corners = np.unique(floor.simplices)
#     point_cloud[floor_corners, 2] = 0
#     try:
#         hull = ConvexHull(point_cloud)
#     except QhullError:
#         print("Failed to construct the hull, offsetting points")
#         print(point_cloud)
#         new_cloud = offset_points(point_cloud)
#         return make_tent(new_cloud)

#     return ConvexHull(point_cloud), floor




if __name__ == "__main__":

    #(0,0,0) - (1,1,1)
    point_cloud = np.random.rand(15,3)
    point_cloud = np.array(
        [
            [0,0,0],
            [1,1,0],
            [0,1,0],
            [1,0,0],
            [1/2, 1/2, 1]
        ]
    )

    tent = Tent(point_cloud)
    print(f"Floor area: {tent.floor_area}\nSurface area: {tent.surface_area}\nVolume: { tent.volume}")
    tent.plot()
