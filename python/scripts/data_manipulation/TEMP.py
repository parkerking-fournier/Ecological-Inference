import numpy as np
import ChaoticModels as cm
import FileIO as io
import csv
import FileIO as io
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.insert(0,'../lib/pyESN')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#__________________________________GenerateData____________________________________
#___________________________________________________________________________________
#___________________________________________________________________________________
#
# This file writes train_x (data) and train_y (labels) csv files
# to be used to train machine learning algorithms. The data being written
# is generated from the 9 most chaotic food web models described as in 
# "Food web complexity and chaotic population dynamics", by Fussmann and Heber (2002).
# 
# The paper can be found here:
#		http://biology.mcgill.ca/faculty/fussmann/articles/FussmannHeber_02ELE.pdf
#
# The free parameters are the mortality rates of species z, y, x, c, and c2, as 
# described in the aformentioned paper. 
#
# Again, as described by Fussman and Heber combination of mortality rates which led to
# one or more species falling below the threshold, e, and therefor did not lead to
# coexistence of species were disregarded. 
#
# Each time series was carried out 50,000 time steps into the future and the first
# 6,000 time steps were omitted from the recorded data. 
#
# @author parkerkingfournier 
#___________________________________________________________________________________
#___________________________________________________________________________________
#___________________________________________________________________________________


#_______________________________________
#______________Generation_______________
#_______________________________________
def gen():

	# Variable Declarations
	length = 400000
	threshold = 4500

	z0 = np.empty((length + 1,))	# five species chain
	y0 = np.empty((length + 1,))
	x0 = np.empty((length + 1,))
	c0 = np.empty((length + 1,))
	p0 = np.empty((length + 1,))

	z1 = np.empty((length + 1,))	# five species chain
	y1 = np.empty((length + 1,))
	x1 = np.empty((length + 1,))
	c1 = np.empty((length + 1,))
	p1 = np.empty((length + 1,))

	z2 = np.empty((length + 1,))	# five species chain
	y2 = np.empty((length + 1,))
	x2 = np.empty((length + 1,))
	c2 = np.empty((length + 1,))
	p2 = np.empty((length + 1,))

	z3 = np.empty((length + 1,))	# five species chain
	y3 = np.empty((length + 1,))
	x3 = np.empty((length + 1,))
	c3 = np.empty((length + 1,))
	p3 = np.empty((length + 1,))

	z4 = np.empty((length + 1,))	# five species chain
	y4 = np.empty((length + 1,))
	x4 = np.empty((length + 1,))
	c4 = np.empty((length + 1,))
	p4 = np.empty((length + 1,))

	z5 = np.empty((length + 1,))	# five species chain
	y5 = np.empty((length + 1,))
	x5 = np.empty((length + 1,))
	c5 = np.empty((length + 1,))
	p5 = np.empty((length + 1,))

	z6 = np.empty((length + 1,))	# five species chain
	y6 = np.empty((length + 1,))
	x6 = np.empty((length + 1,))
	c6 = np.empty((length + 1,))
	p6 = np.empty((length + 1,))

	z7 = np.empty((length + 1,))	# five species chain
	y7 = np.empty((length + 1,))
	x7 = np.empty((length + 1,))
	c7 = np.empty((length + 1,))
	p7 = np.empty((length + 1,))

	e = 0.000000001
	dt = 0.01

	d_z = 0.01	#Found through trial and error that all parameter combinations leading to coexistance had d_z = 0.01
	d_y = 0.01
	d_x = 0.01
	d_c = 0.15

	# Initial Conditions for the food web topologies labeled 0 through 8. All start at 0.5
	z0[0], y0[0], x0[0], c0[0], p0[0] =(
	0.350, 
	0.350, 
	0.350, 
	0.350, 
	0.350)
	

		# five species chain
	z1[0], y1[0], x1[0], c1[0], p1[0] = (0.500025, 0.500025, 0.500025, 0.500025, 0.500025) 	# five species chain
	z2[0], y2[0], x2[0], c2[0], p2[0] = (0.50005, 0.50005, 0.50005, 0.50005, 0.50005) 	# five species chain
	z3[0], y3[0], x3[0], c3[0], p3[0] = (0.500075, 0.500075, 0.500075, 0.500075, 0.500075) 	# five species chain
	z4[0], y4[0], x4[0], c4[0], p4[0] = (0.50001, 0.50001, 0.50001, 0.50001, 0.50001) 	# five species chain
	z5[0], y5[0], x5[0], c5[0], p5[0] = (0.5000125, 0.5000125, 0.5000125, 0.5000125, 0.5000125) 	# five species chain
	z6[0], y6[0], x6[0], c6[0], p6[0] = (0.500015, 0.500015, 0.500015, 0.500015, 0.500015) 	# five species chain
	z7[0], y7[0], x7[0], c7[0], p7[0] = (0.5000175, 0.5000175, 0.5000175, 0.5000175, 0.5000175) 	# five species chain



	# Evaulate time series
	for i in range(length): 
		z0[i+1], y0[i+1], x0[i+1], c0[i+1], p0[i+1] = cm.fiveSpeciesChain(z0[i], y0[i], x0[i], c0[i], p0[i], d_z, d_y, d_x, d_c, dt)
		#z1[i+1], y1[i+1], x1[i+1], c1[i+1], p1[i+1] = cm.fiveSpeciesChain(z1[i], y1[i], x1[i], c1[i], p1[i], d_z, d_y, d_x, d_c, dt)
		#z2[i+1], y2[i+1], x2[i+1], c2[i+1], p2[i+1] = cm.fiveSpeciesChain(z2[i], y2[i], x2[i], c2[i], p2[i], d_z, d_y, d_x, d_c, dt)
		#z3[i+1], y3[i+1], x3[i+1], c3[i+1], p3[i+1] = cm.fiveSpeciesChain(z3[i], y3[i], x3[i], c3[i], p3[i], d_z, d_y, d_x, d_c, dt)
		#z4[i+1], y4[i+1], x4[i+1], c4[i+1], p4[i+1] = cm.fiveSpeciesChain(z4[i], y4[i], x4[i], c4[i], p4[i], d_z, d_y, d_x, d_c, dt)
		#z5[i+1], y5[i+1], x5[i+1], c5[i+1], p5[i+1] = cm.fiveSpeciesChain(z5[i], y5[i], x5[i], c5[i], p5[i], d_z, d_y, d_x, d_c, dt)
		#z6[i+1], y6[i+1], x6[i+1], c6[i+1], p6[i+1] = cm.fiveSpeciesChain(z6[i], y6[i], x6[i], c6[i], p6[i], d_z, d_y, d_x, d_c, dt)
		#z7[i+1], y7[i+1], x7[i+1], c7[i+1], p7[i+1] = cm.fiveSpeciesChain(z7[i], y7[i], x7[i], c7[i], p7[i], d_z, d_y, d_x, d_c, dt)
	#end i
					
	# Check for coexistence and record the data if the species coexist		
	if(z0[length] > e and y0[length] > e and x0[length] > e and c0[length] > e and p0[length] > e):
		# Setting up figure
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		plt.plot(z0[threshold:], y0[threshold:], x0[threshold:], 'K', lw=0.5)
		#plt.plot(z1[threshold:], y1[threshold:], x1[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z2[threshold:], y2[threshold:], x2[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z3[threshold:], y3[threshold:], x3[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z4[threshold:], y4[threshold:], x4[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z5[threshold:], y5[threshold:], x5[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z6[threshold:], y6[threshold:], x6[threshold:], 'tab:purple', lw=0.2)
		#plt.plot(z7[threshold:], y7[threshold:], x7[threshold:], 'tab:purple', lw=0.2)
		plt.show()

# end gen()

gen()




























