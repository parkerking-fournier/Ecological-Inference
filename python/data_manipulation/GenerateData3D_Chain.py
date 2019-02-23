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

	y1 = np.empty((step_count + 1,))
	x1 = np.empty((step_count + 1,))	# tritrophic no omnivory
	c1 = np.empty((step_count + 1,))
	p1 = np.empty((step_count + 1,))

	e = 0.000000001
	dt = 0.01

	min_y, max_y = (0.01, 0.14)
	min_x, max_x = (0.01, 0.09)
	min_c, max_c = (0.01, 0.19)

	grid_fineness = 7

	count = 0

	# Initial Conditions for the food web topologies labeled 0 through 8. All start at 0.5	
	y1[0], x1[0], c1[0], p1[0] = (0.5, 0.5, 0.5, 0.5)		# tritrophic no omnivory

	# Iterate through mortality rates
	for d_y in np.linspace(min_y, max_y, grid_fineness):
		for d_x in np.linspace(min_x, max_x, grid_fineness):
			for d_c in np.linspace(min_c, max_c, grid_fineness):
						
				# Evaulate time series
				for i in range(step_count): 
					y1[i+1], x1[i+1], c1[i+1], p1[i+1] 	= cm.fourSpeciesChain(y1[i], x1[i], c1[i], p1[i], d_y, d_x, d_c, dt)
						
				# Check for coexistence and record the data if the species coexist		
				if(y1[step_count] > e and x1[step_count] > e and c1[step_count] > e and p1[step_count] > e):
					M = np.stack((y1,x1,c1,p1,p1))
					P = np.array([d_y, d_x, d_c])

					if count<4:
						io.writeData(train_x, train_y, M, '1', threshold)
						count = count+1
					else:
						io.writeData(test_x, test_y, M, '1', threshold)
						count = 0
		
					io.writeParameters(parameters, P)
			#d_c
		#d_x
	#d_y
# end gen()





























