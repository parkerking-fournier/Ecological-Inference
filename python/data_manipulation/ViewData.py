import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file = "../../../datasets/tmp/test_x_5chain.npy"

data_matrix = np.load(file)

z = data_matrix[0]
y = data_matrix[1]
x = data_matrix[2]
c = data_matrix[3]
p = data_matrix[4]

# Setting up figure
fig = plt.figure()
ax = fig.gca(projection='3d')
		
plt.plot(z, y, x)

plt.show()