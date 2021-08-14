from modules.GeometryDesign.tent import Tent
import numpy as np

def run_example():
    # Pyramid example
    print("Making a pyramid")
    # Define the points first
    pyramid_points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0], # floor corners
        [.5, .5, 1] # pyramidion / capstone
    ])

    # Instansiate the object
    pyramid = Tent(pyramid_points)

    # Print some information about the pyramid
    print(f"Pyramid\n{'-'*50}\nFloor area: {pyramid.floor_area}\nSurface area: {pyramid.surface_area}\nVolume: {pyramid.volume}\n\n")

    # Plot the pyramid
    pyramid.plot()



    # Box example:
    print(f"{'-'*50}\nMaking a box")
    # Define the points
    box_points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [1,1,0], # floor corners
        [0,0,1], [1,0,1], [0,1,1], [1,1,1] # Ceiling/roof corners
    ])

    # Instansiate the object
    box = Tent(box_points)

    # Print the volume
    print(f"Box volume: {box.volume}\n\n")

    # Plot the box
    box.plot()


    # Caveats

    # Caveat 1

    # as the furthermost points (x, y axis) are projected down to z = 0
    # defining a box with floor points higher z = 0 will results in the same box 
    # with floor defined to z = 0 
    box_points2 = np.array([
        [0,0,.5], [1,0,.5], [0,1,.5], [1,1,.5], # floor corners at z = 0.5
        [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    ])

    box2 = Tent(box_points2)
    # compare
    print("comparing two boxes with different point clouds")
    print(f"Volume is {'same' if box2.volume == box.volume else 'not same'}")
    print(f"Surface area is {'same' if box2.surface_area == box.surface_area else 'not same'}\n\n")



    # Caveat 2

    # If qhull fails to construct the convexhull the points will be given an offset and 
    # after that we try again (max 10 times).
    offset_points = np.array([[.5, .5, .5]] * 10) # 10 points all at (.5, .5, .5)
    print("Constructing a tent with bad points:\n")
    t = Tent(offset_points)

run_example() # You can comment this line out and do your own things