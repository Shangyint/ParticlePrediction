from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# current_data_dir = 'test_0'
os.chdir('../model/data')
plot_all = True
start = 4000
end = 9000

fig = plt.figure()
ax = plt.axes(projection='3d')
df = pd.read_csv('test_y.csv')
if plot_all:
    sphere_info = df.values
else:
    sphere_info = df.values[start:end, :]
ax.plot3D(sphere_info[:, 0], sphere_info[:, 1], sphere_info[:, 2], 'green', label='Observation')
ax.legend()

df = pd.read_csv('yhat_2_epoch_200.csv')
if plot_all:
    sphere_info = df.values
else:
    sphere_info = df.values[start:end, :]
ax.plot3D(sphere_info[:, 0], sphere_info[:, 1], sphere_info[:, 2], 'gray', label='Prediction 2')
ax.legend()

df = pd.read_csv('yhat_5_epoch_200.csv')
if plot_all:
    sphere_info = df.values
else:
    sphere_info = df.values[start:end, :]
ax.plot3D(sphere_info[:, 0], sphere_info[:, 1], sphere_info[:, 2], 'red', label='Prediction 5')
ax.legend()

plt.show()

# back to original directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))