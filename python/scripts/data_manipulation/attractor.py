import numpy as np
import ChaoticModels as cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Variable Declarations
length     = 99000
threshold  = 65000
num_points = 300
seperation = 0.001
dt  = 0.01
d_z = 0.01
d_y = 0.01
d_x = 0.01
d_c = 0.11

# Creating the 3d numpy array
data = np.zeros([num_points, 5,length])

# Initial Conditions
for i in range(num_points):
	for j in range(5):
		if j==0:
			data[i,j,0] = 0.5 + i*seperation
		else:
			data[i,j,0] = 0.5
# Evaulate time series
for i in range(num_points):
	for k in range(1,length): 
		data[i,0,k], data[i,1,k], data[i,2,k], data[i,3,k], data[i,4,k] = cm.fiveSpeciesChain(data[i,0,(k-1)], data[i,1,(k-1)], data[i,2,(k-1)], data[i,3,(k-1)], data[i,4,(k-1)], d_z, d_y, d_x, d_c, dt)

# Set up figure
fig = plt.figure()
ax = fig.gca(projection='3d')
for i in range(num_points):
	plt.plot(data[i,0][threshold:], data[i,1][threshold:], data[i,2][threshold:], 'B', lw=0.4)
plt.show()