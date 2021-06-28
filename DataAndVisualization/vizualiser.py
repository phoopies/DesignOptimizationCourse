import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting as pdplt
from plotly.express import parallel_coordinates, colors

# Filename without extension
def plot_df(df):
    k = df.size - 1
    if k == 2: # Scatterplot
        plt.scatter(df[:,0], df[:,1])
        plt.show()
    else:
        fig = parallel_coordinates(df.sample(frac = 1), dimensions=df.columns, color="Names")
        fig.show()


def to_df(arr):
    names = [255] * arr.shape[0]
    df = pd.DataFrame(arr)
    df['Names'] = names
    return df

def get_data(filename):
    d = np.load(f"DataAndVisualization/{filename}.npz")
    return d['obj'], d['var']

obj, var = get_data("gd_surface_volume__floor")
# Scale values to [0,1]. TODO 1 should be ideal
df = to_df(obj)
# nadir = np.atleast_2d(nadir)
# ndf = pd.DataFrame(nadir)
# ndf['Names'] = 2

# ideal = np.atleast_2d(ideal)
# idf = pd.DataFrame(ideal)
# idf['Names'] = 12
# df = df.append(idf, ignore_index=True)
# df = df.append(ndf, ignore_index=True)
# print(df)
# df[df.columns[:-1]] -= nadir
# df[df.columns[:-1]] /= ideal
plot_df(df)