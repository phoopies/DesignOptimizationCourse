from  modules.DataAndVisualization.vizualiser import (
    plot_scatter as scatter,
    visualize as interactive_scatter,
    plot_parallel as parallel,
)
from modules.utils import load

# Visualizing the geometry design problem with constant floor

# First load the solution. For more details on on the solution check the readme file in modules/DataAndVisualization
obj, var, nadir, ideal = load("gdcExample1")

# You can ignore this part
# Make sure all values are positive as some of the objectives may be flipped
# Because setting pfront to true in creating problem method will flip some values.
obj = abs(obj) 

# Axis names for the plot
axis_names = ["Surface area", "Volume"]

# Constant floor Geometry design problems supports interactive scatter plot as it is only 2 dimensional:
# Click a point => Corresponging tent will be plotted and values will be printed to console 
# => close the tent plot => back to objectives plot
interactive_scatter(obj, var, axis_names)



# You can also use other plots:

# Scatter
scatter(obj, axis_names)

# parallel, this is actually quite interesting
parallel(obj, axis_names)
