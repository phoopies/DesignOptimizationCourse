from math import floor
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

class Tent:
    _point_cloud: np.ndarray
    main_hull: ConvexHull
    floor_hull: ConvexHull

    def __init__(self, point_cloud) -> None:
        self._point_cloud = point_cloud
        self.main_hull, self.floor_hull = make_tent(point_cloud)

    @property
    def floor_area(self):
        return self.floor_hull.volume

    @property
    def surface_area(self):
        return self.main_hull.area - self.floor_area

    @property
    def volume(self):
        return self.main_hull.volume
    
    def plot(self):
        ax = plt.axes(projection='3d')
        x,y,z = np.split(point_cloud, 3, 1)
        ax.scatter3D(x,y,z)

        #Plotting the hull
        for s in self.main_hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(self._point_cloud[s, 0], self._point_cloud[s, 1], self._point_cloud[s, 2], "r-")
        
        for v in self.floor_hull.simplices:
            v = np.append(v, v[0])
            ax.plot(self._point_cloud[v, 0], self._point_cloud[v, 1])
        plt.show()

def floor_hull(point_cloud) -> ConvexHull:
    """
    """
    point_cloud_2d = np.delete(point_cloud,2,1) # Delete z axis
    floor_hull = ConvexHull(point_cloud_2d) # Make the 2d hull
    return floor_hull

# Better name!
def make_tent(point_cloud):
    """
    Constructs a 2d convex hull from x,y points and projects them to floor lever z = 0
    Then constructs a convex hull from new points (with the floor)

    Returns:
        ConvexHull of 3d point cloud and the floor "hull"
    """
    floor = floor_hull(point_cloud)
    floor_corners = np.unique(floor.simplices)
    point_cloud[floor_corners, 2] = 0
    return ConvexHull(point_cloud), floor

if __name__ == "__main__":

    #(0,0,0) - (1,1,1)
    point_cloud = np.random.rand(15,3)

    tent = Tent(point_cloud)
    print(f"Floor area: {tent.floor_area}\nSurface area: {tent.surface_area}\nVolume: { tent.volume}")
    tent.plot()