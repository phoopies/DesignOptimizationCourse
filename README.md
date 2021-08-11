# JSS30 Two bar truss and geometry design problems with DESDEO framework

You can find all necessary python files from the root folder.
Information on precalculated solutions can be found from [HERE](modules/DataAndVisualization/README.md).

## Python files

### Creating the problems 
One can use these files to create DESDEOs MOProblem instances of two bar truss problem or either one of the geometry design problem.
You can find examples on creating the problem from the file itself. 

| file name                                | Details             |
| -------------                            | -------------       |
| createTwoBartrussProblem.py              | Creating a two bar truss problem |
| createGeometryDesignProblem.py           | Creating the default geometry design problem |
| createFloorGeometryDesingProblem.py      | Creating the 2 dimensional (surface area and volume) geometry design problem with constant floor | 

### Visualizing the solutions
For the geometry design problems (< 4d) one can use interactive scatter plots which will scatter the objectives values
and by clicking any point from the scatter plot a tent representation will be plotted of the corresponding decision variables.

Examples of both scatter plots and parallel plots can be found from the files

| file name                                | Details             |
| -------------                            | -------------       |
| visualizeTwoBarTruss.py                  | Visualizing the two bar truss problem |
| visualizeGeometryProblem.py              | Visualizing the default geometry design problem |
| visualizeFloorGeometryProblem.py         | Visualizing the geometry design problem with constant floor with interactive scatter plot|


### Making tents
The tentExample.py file has a few examples on constructing Tents and how to use them. Also a few mentions on some caveats.