from  modules.DataAndVisualization.vizualiser import (
    visualize as interactive_scatter,
    plot_parallel as parallel,
)
from modules.utils import load

# Visualizing the geometry design problem

# Objectives as a reminder
# surface area, volume, min height and floor area

# First load the solution. For more details on on the solution check the readme file in modules/DataAndVisualization
obj, var, _nadir, _ideal = load("gd1")

# You can ignore this part
# Make sure all values are positive as some of the objectives may be flipped
# Because setting pfront to true in creating problem method will flip some values.
obj = abs(obj)

# Axis names for the plot
axis_names = ["Surface area", "Volume", "Min height", "Floor area"]

# As this is a 4d solution we'll have to use a parallel plot:
parallel(obj, axis_names)


# Geometry design problems supports 2 and 3 dimensional interactive scatter plots:
# Click a point => Corresponging tent will be plotted and values will be printed to console 
# => close the tent plot => back to objectives plot

# Load a 3d problem
obj, var, _nadir, _ideal = load("gd2")
obj = abs(obj)
axis_names = ["Surface area", "Volume", "floor area"]

interactive_scatter(obj, var, axis_names)
