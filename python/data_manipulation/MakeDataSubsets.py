import GenerateData2D_Chain 	as g2_chain
import GenerateData2D_A 		as g2_a
import GenerateData3D_B 		as g3_b
import GenerateData3D_C 		as g3_c
import GenerateData3D_Chain 	as g3_chain
import GenerateData3D_No 		as g3_n
import GenerateData3D_Partial 	as g3_p
import GenerateData4D 			as g4

import numpy 	as np
import pandas 	as pd

#__________________________________MakeData_________________________________________
#___________________________________________________________________________________
#___________________________________________________________________________________
#
#
# CAREFUL!!! THIS FILE WILL MAKE YOUR COMPUTER HOT AND TAKE THE BETTER PART OF
# AN HOUR TO RUN.
#
#
# This file writes train_x (data) and train_y (labels) csv files
# to be used to train machine learning algorithms. The csv files are then stored
# as numpy arrays (.npy file extension) so they can be accessed more efficiently
# The data being written is generated from the 9 most chaotic food web models 
# described as in "Food web complexity and chaotic population dynamics", by Fussmann 
# and Heber (2002). The paper can be found here:
#
#		http://biology.mcgill.ca/faculty/fussmann/articles/FussmannHeber_02ELE.pdf
#
# The free parameters are the mortality rates of species z, y, x, c1, and c2, as 
# described in the aformentioned paper. 
#
# Again, as described by Fussman and Heber combination of mortality rates which led to
# one or more species falling below the threshold, e, and therefor did not lead to
# coexistence of species were disregarded. 
#
# Each time series was carried out 200,000 time steps into the future and the first
# 30,000 time steps were omitted from the recorded data. 
#
# @author parkerkingfournier 
#___________________________________________________________________________________
#___________________________________________________________________________________
#___________________________________________________________________________________

#_______________________________________
#______________Main Method______________
#_______________________________________
def main():

	# Length describes ow far out to run each model in the future 
	# cutoff is the amount of time steps omitted from the beginning 
	# of each model
	# Each data length will be 200,000 + 1 - 50232 = 149769 = 387*387
	length = 200000
	cutoff =  50232

	# File Names
	train_x 	= '../../../datasets/train_x_5chain_tmp.csv'
	train_y 	= '../../../datasets/train_y_5chain_tmp.csv'
	test_x 		= '../../../datasets/test_x_5chain_tmp.csv'
	test_y 		= '../../../datasets/test_y_5chain_tmp.csv'
	
	train_x_pkl	= '../../../datasets/train_x_5chain_tmp.npy'
	train_y_pkl = '../../../datasets/train_y_5chain_tmp.npy'
	test_x_pkl 	= '../../../datasets/test_x_5chain_tmp.npy'
	test_y_pkl 	= '../../../datasets/test_y_5chain_tmp.npy'

	#p0 = '../../../population_models/parameters_chains/0.csv'
	#p1 = '../../../population_models/parameters_chains/1.csv'
	#p2 = '../../../population_models/parameters_chains/2.csv'
	#p3 = '../../../population_models/parameters_chains/3.csv'
	#p4 = '../../../population_models/parameters_chains/4.csv'
	#p5 = '../../../population_models/parameters_chains/5.csv'
	#p6 = '../../../population_models/parameters_chains/6.csv'
	p7 = '../../../population_models/parameters_chains/HERE.csv'
	#p8 = '../../../population_models/parameters_chains/8.csv'

	# Generate and write data to csv files
	#print "\nGenerating data for..."
	#print"			3 species chain"
	#g2_chain.gen(	train_x, train_y, test_x, test_y, p0, length, cutoff)
	#print "			tritrophic omnivory a"
	#g2_a.gen(		train_x, train_y, test_x, test_y, p1, length, cutoff)
	#print "			tritrophic omnivory B"
	#g3_b.gen(		train_x, train_y, test_x, test_y, p2, length, cutoff)
	#print "			tritrophic omnivory C"
	#g3_c.gen(		train_x, train_y, test_x, test_y, p3, length, cutoff)
	#print "			4 species chain"
	#g3_chain.gen(	train_x, train_y, test_x, test_y, p4, length, cutoff)
	#print "			tritrophic no omnivory"
	#g3_n.gen(		train_x, train_y, test_x, test_y, p5, length, cutoff)
	#print "			tritrophic partial omnivory"
	#g3_p.gen(		train_x, train_y, test_x, test_y, p6, length, cutoff)
	print "			5 species chain"
	g4.gen(			train_x, train_y, test_x, test_y, p7, length, cutoff)
	#print "\n	Finished!"
	#print "	Training data written to ", train_x, " and labels written to ", train_y
	#print "	Test data written to ", test_x, " and labels written to ", test_y
	
	# Store data as .npy files
	print "\nLoading files to pickle..."
	print "		Loading training sets..."
	X_TRAIN = pd.read_csv(train_x, 	sep=',', header=None)
	Y_TRAIN = pd.read_csv(train_y, 	sep=',', header=None)
	print "		Loading test sets..."
	X_TEST 	= pd.read_csv(test_x, 	sep=',', header=None)
	Y_TEST 	= pd.read_csv(test_y, 	sep=',', header=None)
	print "	Finished!"
	
	print "Pickling files..."
	np.save(train_x_pkl, 	X_TRAIN.values, allow_pickle=True)
	np.save(train_y_pkl, 	Y_TRAIN.values, allow_pickle=True)
	np.save(test_x_pkl, 	X_TEST.values, 	allow_pickle=True)
	np.save(test_y_pkl, 	Y_TEST.values, 	allow_pickle=True)
	print "	Finished!"
	print "	Training data pickled to ", train_x_pkl, " and labels pickled to ", train_y_pkl
	print "	Test data pickled to ", test_x_pkl, " and labels pickled to	", test_y_pkl

	print "\nFor details on the models used consult the README file located in the 'doc' directory.\n"

main()