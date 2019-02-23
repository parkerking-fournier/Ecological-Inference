import numpy as np
import ChaoticModels as cm
import FileIO as io

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
# The free parameters are the mortality rates of species z, y, x, c1, and c2, as 
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
	
	z8 = np.empty((step_count + 1,))	# quintrophic omnivory
	y8 = np.empty((step_count + 1,))
	x8 = np.empty((step_count + 1,))
	c8_1 = np.empty((step_count + 1,))
	c8_2 = np.empty((step_count + 1,))
	p8_1 = np.empty((step_count + 1,))
	p8_2 = np.empty((step_count + 1,))
	p8_3 = np.empty((step_count + 1,))

	e = 0.000000001
	dt = 0.01

	d_c1 = 0.25
	d_c2 = 0.25
	d_x = 0.1075
	d_z = 0.06
	min_y, max_y = (0.007, 0.013)

	grid_fineness = 500

	count = 0

	# Initial Conditions for the food web topologies labeled 0 through 8. All start at 0.5
	z8[0], y8[0], x8[0], c8_1[0], c8_2[0], p8_1[0], p8_2[0], p8_3[0] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5) # quintrophic omnivory

	# Iterate through mortality rates
	for d_y in np.linspace(min_y, max_y, grid_fineness):

		print d_z, ",", d_y, ",", d_x, ",", d_c1, ",", d_c2

		# Evaulate time series
		for i in range(step_count):
			z8[i+1], y8[i+1], x8[i+1], c8_1[i+1], c8_2[i+1], p8_1[i+1], p8_2[i+1], p8_3[i+1]  = cm.quintrophicOmnivory(z8[i], y8[i], x8[i], c8_1[i], c8_2[i], p8_1[i], p8_2[i], p8_3[i], d_z, d_y, d_x, d_c1, d_c2, dt)
		
		# Check for coexistence and record the data if the species coexist
		if(z8[step_count] > e and y8[step_count] > e and x8[step_count] > e and c8_1[step_count] > e and c8_2[step_count] > e and p8_1[step_count] > e and p8_2[step_count] > e and p8_3[step_count] > e):
			M = np.stack((z8,y8,x8,c8_1,c8_2,p8_1,p8_2,p8_3))
			P = np.array([d_z, d_y, d_x, d_c1, d_c2])

			if count<4:
				io.writeData(train_x, train_y, M, '8', threshold)
				count = count+1
			else:
				io.writeData(test_x, test_y, M, '8', threshold)
				count = 0

			io.writeParameters(parameters, P)
	# d_y
# end gen()





























