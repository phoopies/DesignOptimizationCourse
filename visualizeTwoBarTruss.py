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
obj, var, nadir, ideal = load("tb2")

# This problem is 2 dimensional as it is only optimizing weight and stress
# so we can use a scatter or a parallel plot to visualize it. We will use a scatter plot

# Set the axis names accordingly
axis_names = ["weight", "Stress"]

# Plot
scatter(obj, axis_names)


# An example on parallel plots

# Parallel plot will open a new browser window and display the plot there. 
# Only 1000 random samples are chosen for the plot
# You can choose an axis range to highlight solutions that fall in that range interactively

# Load the problem
obj2, var2, nadir2, ideal2 = load("tbExample1")

# Set axis names
axis_names = ["weight", "stress", "buckling stress", "deflection"]

# Plot
parallel(obj2, axis_names)