import numpy as np
import matplotlib.pyplot as plt
from plotly.express import parallel_coordinates
import matplotlib.pyplot as plt
import numpy as np
from modules.GeometryDesign.tent import Tent

exit_program = True


def point_cloud_1d_to_3d(point_cloud_1d: np.ndarray):
    points_n = int(point_cloud_1d.size/3)
    point_cloud_3d = point_cloud_1d.reshape(points_n, 3)
    return point_cloud_3d

def plot_parallel(obj, axis_names = None):
    sample = obj
    if obj.shape[0] > 1000: 
        sample = obj[np.random.choice(obj.shape[0], 1000, replace=False)]
    fig = parallel_coordinates(sample, labels=axis_names)
    fig.show()

def plot_scatter_clickable(obj, var, axis_names = None, axis_ranges = None):
    global exit_program
    exit_program = True
    k = obj.shape[1]
    fig, ax = plt.subplots()
    ax.set_title("Pareto front")
    if k == 2:
        ax.plot(obj[:,0], obj[:,1], marker='o', picker=True, pickradius = 5,linestyle = 'None')
        if axis_names is not None and len(axis_names) == 2:
            plt.xlabel(axis_names[0])
            plt.ylabel(axis_names[1])
        fig.canvas.mpl_connect('pick_event', lambda e: onpick(e, var, obj,False))
    elif k == 3:
        ax = fig.add_subplot(projection='3d')
        ax.plot(obj[:,0], obj[:,1], obj[:,2], marker='o', picker=True, pickradius = 5, linestyle = 'None')
        if axis_names is not None and len(axis_names) == 3:
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            ax.set_zlabel(axis_names[2])
        fig.canvas.mpl_connect('pick_event', lambda e:onpick(e, var, obj, True))
    else:
        raise Exception("Wrong dimension count")
    if axis_ranges is not None and axis_ranges.shape[0] == 2:
        plt.xlim(axis_ranges[0])
        plt.ylim(axis_ranges[1])
        if k == 3:
            plt.zlim(axis_ranges[2])
    plt.show()

def plot_scatter(df, axis_names = None):
    k = df.shape[1]
    fig, ax = plt.subplots()
    ax.set_title("Pareto front")
    if k == 2:
        ax.plot(df[:,0], df[:,1], marker='o', linestyle = 'None')
        if axis_names is not None and len(axis_names) == 2:
            plt.xlabel(axis_names[0])
            plt.ylabel(axis_names[1])
    elif k == 3:
        ax = fig.add_subplot(projection='3d')
        ax.plot(df[:,0], df[:,1], df[:,2], marker='o', linestyle = 'None')
        if axis_names is not None and len(axis_names) == 3:
            ax.set_xlabel(axis_names[0])
            ax.set_ylabel(axis_names[1])
            ax.set_zlabel(axis_names[2])
    else:
        raise Exception("Wrong dimension count")
    plt.show()
    
    
def onpick(event, var, obj, multi: bool = False):
    global exit_program
    exit_program = False
    thisline = event.artist
    if multi: xdata, ydata ,zdata = thisline.get_data_3d()
    else:     
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
    ind = event.ind[0]
    point = [xdata[ind], ydata[ind]]
    if multi: point.append(zdata[ind])
    print('Objectives:', point)
    p = var[np.where(np.all(point == obj, axis=1))]
    p = p[0]
    print('Decision variable:', p)
    # plt.close()
    t = Tent(point_cloud_1d_to_3d(p))
    print(t.surface_area, t.volume)
    t.plot()

def visualize(obj, var, axis_names = None):
    global exit_program
    exit_program = False
    try:
        while True:
            if not exit_program:
                plot_scatter_clickable(obj, var, axis_names)
            else: break
    except KeyboardInterrupt:
        print("OK")
