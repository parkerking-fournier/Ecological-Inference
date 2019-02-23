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
	
	x6 = np.empty((step_count + 1,))	# tritrophic partial omnivory
	c6_1 = np.empty((step_count + 1,))
	c6_2 = np.empty((step_count + 1,))
	p6_1 = np.empty((step_count + 1,))
	p6_2 = np.empty((step_count + 1,))

	e = 0.000000001
	dt = 0.01

	min_x, max_x = (0.17, 0.39)
	min_c1, max_c1 = (0.3, 0.87)
	min_c2, max_c2 = (0.01, 0.27)

	grid_fineness = 7

	count = 0

	# Initial Conditions for the food web topologies labeled 0 through 8. All start at 0.5	
	x6[0], c6_1[0], c6_2[0], p6_1[0], p6_2[0] = (0.5, 0.5, 0.5, 0.5, 0.5)		# tritrophic partial omnivory

	# Iterate through mortality rates
	for d_x in np.linspace(min_x, max_x, grid_fineness):
		for d_c1 in np.linspace(min_c1, max_c1, grid_fineness):
			for d_c2 in np.linspace(min_c2, max_c2, grid_fineness):
						
				# Evaulate time series
				for i in range(step_count): 
					x6[i+1], c6_1[i+1], c6_2[i+1], p6_1[i+1], p6_2[i+1] 	= cm.tritrophicPartialOmnivory(x6[i], c6_1[i], c6_2[i], p6_1[i], p6_2[i], d_x, d_c1, d_c2, dt)
						
				# Check for coexistence and record the data if the species coexist		
				if(x6[step_count] > e and c6_1[step_count] > e and c6_2[step_count] > e and p6_1[step_count] > e and p6_2[step_count] > e):
					M = np.stack((x6,c6_1,c6_2,p6_1,p6_2))
					P = np.array([d_x, d_c1, d_c2])

					if count<4:
						io.writeData(train_x, train_y, M, '6', threshold)
						count = count+1
					else:
						io.writeData(test_x, test_y, M, '6', threshold)
						count = 0
		
					io.writeParameters(parameters, P)
			#d_c2			
		#d_c1
	#d_x
# end gen()