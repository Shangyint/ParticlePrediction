from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

current_data_dir = 'test_0'
os.chdir('../data/' + current_data_dir)

fig = plt.figure()
ax = plt.axes(projection='3d')
df = pd.read_csv('sphere1.csv')
sphere_info = df.values
ax.plot3D(sphere_info[:, 0], sphere_info[:, 1], sphere_info[:, 2], 'gray')
df = pd.read_csv('sphere2.csv')
sphere_info = df.values
ax.plot3D(sphere_info[:, 0], sphere_info[:, 1], sphere_info[:, 2], 'green')
plt.show()

# back to original directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))