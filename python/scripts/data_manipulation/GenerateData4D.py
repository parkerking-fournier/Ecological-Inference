import numpy as np
import ChaoticModels as cm
import FileIO as io
import csv

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
def gen(train_x, train_y, test_x, test_y, parameters, length, cutoff):

	# Variable Declarations
	step_count = length
	threshold = cutoff

	z0 = np.empty((step_count + 1,))	# five species chain
	y0 = np.empty((step_count + 1,))
	x0 = np.empty((step_count + 1,))
	c0 = np.empty((step_count + 1,))
	p0 = np.empty((step_count + 1,))

	e = 0.000000001
	dt = 0.01

	d_z = 0.01	#Found through trial and error that all parameter combinations leading to coexistance had d_z = 0.01
	min_y, max_y = (0.01, 0.09)
	min_x, max_x = (0.01, 0.11)
	min_c, max_c = (0.01, 0.11)

	grid_fineness = 7

	count = 0

	# Initial Conditions for the food web topologies labeled 0 through 8. All start at 0.5
	z0[0], y0[0], x0[0], c0[0], p0[0] = (0.5, 0.5, 0.5, 0.5, 0.5) 	# five species chain

	# Iterate through mortality rates
	for d_y in np.linspace(min_y, max_y, grid_fineness):
		for d_x in np.linspace(min_x, max_x, grid_fineness):
			for d_c in np.linspace(min_c, max_c, grid_fineness):

				# Evaulate time series
				for i in range(step_count): 
					z0[i+1], y0[i+1], x0[i+1], c0[i+1], p0[i+1] = cm.fiveSpeciesChain(z0[i], y0[i], x0[i], c0[i], p0[i], d_z, d_y, d_x, d_c, dt)
					
				# Check for coexistence and record the data if the species coexist		
				if(z0[step_count] > e and y0[step_count] > e and x0[step_count] > e and c0[step_count] > e and p0[step_count] > e):
					M = np.stack((z0,y0,x0,c0,p0))
					P = np.array([d_z, d_y, d_x, d_c])
					
					if count<4:
						io.writeData(train_x, train_y, M, '0', threshold)
						count = count+1
					else:
						io.writeData(test_x, test_y, M, '0', threshold)
						count = 0
						io.writeParameters(parameters, P)

				#d_c
			#d_x
		# d_y
	# d_z
# end gen()





























