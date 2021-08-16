from  modules.DataAndVisualization.vizualiser import (
    plot_scatter as scatter,
    plot_parallel as parallel,
)
from modules.utils import load
# Visualizing the two bar truss problem

# You can use the parallel plot with any dimension but the scatter will only work with 2 and 3 dimension.

# objectives as a reminder:
# weight, stress, buckling stress and deflection

# First load the solution. For more details on on the solution check the readme file in modules/DataAndVisualization
obj, var, nadir, ideal = load("tb4")

# This problem is 3 dimensional as it is only optimizing weight, stress and buckling stress
# so we can use a scatter or a parallel plot to visualize it. We will use a scatter plot

# Set the axis names accordingly
axis_names = ["weight", "Stress", "buckling stress"]

# (3D) Scatter plot
scatter(obj, axis_names)


# An example on parallel plots

# Parallel plot will open a new browser window and display the plot there. 
# Only 1000 random samples are chosen for the plot
# You can choose an axis ranges to highlight solutions that fall in those ranges interactively

# Load the problem
obj2, var2, nadir2, ideal2 = load("tb1")

# Set axis names
axis_names = ["weight", "stress", "buckling stress", "deflection"]

# Set dimensions for each axis, rows should match objectives.
# (This will filter objectives with values that break the dimensions thus they wont be shown on the plot. None will be converted to -/+ np.inf)
dimensions = [
    [10,None],
    [0,5],
    [None,None],
    [0,1]
]

# Plot
parallel(obj2, axis_names, dimensions)