import math
import numpy as np
import EcologicFunctions as ef 

#_______________________________________
#____________Global Variables___________
#_______________________________________

# Variables relating to functional response
a_1 = 0.125 	#Z -> Y
b_1 = 0.25
a_2 = 0.25		#Y -> X
b_2 = 0.5
a_3 = 1.0		#X -> Ci 	(i = 1,2)
b_3 = 2.0
a_4 = 7.5		#C1 -> P1 	&	C2 -> P3
b_4 = 5.0
a_5 = 0.25		#X -> Pi 	(i = 1,2,3)
b_5 = 0.5
a_6 = 2.5		#C1 -> P3	&	C2 -> P1
b_6 = 5.0
a_7 = 5.0		#Ci -> P2 	(i = 1,2)
b_7 = 5.0

# Intrinsic Growth Rate of Primary Producers
r = 2.5

#_______________________________________
#____________Ecologic Models____________
#_______________________________________

# Five spieces food chain
#		
#		Z
#		|
#		Y
#		|
#		X
#		|
#		C
#		|
#		P
#
def fiveSpeciesChain(z, y, x, c, p, d_z, d_y, d_x, d_c, dt):

	z_dot = z*(ef.f(y,y,a_1,b_1) - d_z)
	y_dot = y*(ef.f(x,x,a_2,b_2) - ef.f(z,y,a_1,b_1) - d_y)
	x_dot = x*(ef.f(c,c,a_3,b_3) - ef.f(y,x,a_2,b_2) - d_x)
	c_dot = c*(ef.f(p,p,a_4,b_4) - ef.f(x,c,a_3,b_3) - d_c)
	p_dot = p*(r*(1 - p) 	  	 - ef.f(c,p,a_4,b_4))
	
	if math.isnan(z_dot):
		z_dot = np.nextafter(0,1)
	if math.isnan(y_dot):
		y_dot = np.nextafter(0,1)
	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c_dot):
		c_dot = np.nextafter(0,1)
	if math.isnan(p_dot):
		p_dot = np.nextafter(0,1)

	z_new = z + z_dot*dt
	y_new = y + y_dot*dt
	x_new = x + x_dot*dt
	c_new = c + c_dot*dt
	p_new = p + p_dot*dt

	return z_new, y_new, x_new, c_new, p_new


# Four spieces food chain
#		
#		Y
#		|
#		X
#		|
#		C
#		|
#		P
#
def fourSpeciesChain(y, x, c, p, d_y, d_x, d_c, dt):

	y_dot = y*(ef.f(x,x,a_2,b_2) - d_y)
	x_dot = x*(ef.f(c,c,a_3,b_3) - ef.f(y,x,a_2,b_2) - d_x)
	c_dot = c*(ef.f(p,p,a_4,b_4) - ef.f(x,c,a_3,b_3) - d_c)
	p_dot = p*(r*(1 - p) 	  - ef.f(c,p,a_4,b_4))
	
	if math.isnan(y_dot):
		y_dot = np.nextafter(0,1)
	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c_dot):
		c_dot = np.nextafter(0,1)
	if math.isnan(p_dot):
		p_dot = np.nextafter(0,1)

	y_new = y + y_dot*dt
	x_new = x + x_dot*dt
	c_new = c + c_dot*dt
	p_new = p + p_dot*dt

	return y_new, x_new, c_new, p_new


# Three spieces food chain
#		
#		X
#		|
#		C
#		|
#		P
#
def threeSpeciesChain(x, c, p, d_x, d_c, dt):

	x_dot = x*(ef.f(c,c,a_3,b_3) - d_x)
	c_dot = c*(ef.f(p,p,a_4,b_4) - ef.f(x,c,a_3,b_3) - d_c)
	p_dot = p*(r*(1 - p) 	  - ef.f(c,p,a_4,b_4))
	
	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c_dot):
		c_dot = np.nextafter(0,1)
	if math.isnan(p_dot):
		p_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c_new = c + c_dot*dt
	p_new = p + p_dot*dt

	return x_new, c_new, p_new


# Food web with three trophic levels and an omnivorous predator
#
#			X
#		   /|
#		  / |
#		 C	| 
#		  \ |
#	       \|
#			P
#
def tritrophicOmnivoryA(x, c, p, d_x, d_c, dt):

	x_dot = x*((a_3*c + a_5*p)/(1 + b_3*c + b_5*p) - d_x)
	c_dot = c*(ef.f(p,p,a_4,b_4) - (a_3*x)/(1 + b_3*c + b_5*p) - d_c)
	p_dot = p*(r*(1-p) - ef.f(c,p,a_4,b_4) - (a_5*x)/(1 + b_5*p + b_3*c))

	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c_dot):
		c_dot = np.nextafter(0,1)
	if math.isnan(p_dot):
		p_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c_new = c + c_dot*dt
	p_new = p + p_dot*dt

	return x_new, c_new, p_new


# Food web with three trophic levels and an omnivorous predator
#
#			X
#		   /|\
#		  / | \
#		 C1	|  C2
#		  \ | /
#	       \|/
#			P
#
def tritrophicOmnivoryB(x, c1, c2, p, d_x, d_c1, d_c2, dt):

	x_dot = x*((a_3*c1 + a_3*c2 + a_5*p)/(1 + b_3*c1 + b_3*c2 + b_5*p) - d_x)
	c1_dot = c1*(ef.f(p,p,a_4,b_4) - ef.f(x,c1,a_3,b_3) - d_c1)
	c2_dot = c2*(ef.f(p,p,a_4,b_4) - ef.f(x,c2,a_3,b_3) - d_c2)
	p_dot = p*(r*(1-p) - ef.f(c1,p,a_4,b_4) - ef.f(c2,p,a_4,b_4))

	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c1_dot):
		c1_dot = np.nextafter(0,1)
	if math.isnan(c2_dot):
		c2_dot = np.nextafter(0,1)
	if math.isnan(p_dot):
		p_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c1_new = c1 + c1_dot*dt
	c2_new = c2 + c2_dot*dt
	p_new = p + p_dot*dt

	return x_new, c1_new, c2_new, p_new


# Food web with three trophic levels and an omnivorous predator
#
#	--------X--------
#	|	   / \      |
#	|	  /   \     |
#	|	 /     \    |
#	|  C1       C2  |
#	|  |\	    /|  |
#	|  | \     / |  |
#   |  |  \   /  |  |
#	|  |   \ /	 |  |
#   |  |   / \   |  |
#   |  |  /   \  |  |
#   |  | /     \ |  |
#	|--P1       P2--|
#
def tritrophicOmnivoryC(x, c1, c2, p1, p2, d_x, d_c1, d_c2, dt):

	x_dot = x*((a_3*c1 + a_3*c2 + a_5*p1 + a_5*p2)/(1 + b_3*c1 + b_3*c2 + b_5*p1 + b_5*p1) - d_x)
	c1_dot = c1*((a_4*p1 + a_6*p1)/(1+b_4*p1 + b_6*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2 + b_5*p1 + b_5*p2) - d_c1)
	c2_dot = c2*((a_6*p1 + a_2*p2)/(1+b_6*p1 + b_4*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2 + b_5*p1 + b_5*p2) - d_c2)
	p1_dot = p1*(r*(1-p1) - (a_4*c1)/(1 + b_4*p1 + b_6*p2) - (a_6*c2)/(1 + b_6*p1 + b_4*p2) - (a_5*x)/(1 + b_5*p1 + b_5*p2 + b_3*c1 + b_3*c2))
	p2_dot = p2*(r*(1-p2) - (a_6*c1)/(1 + b_6*p1 + b_4*p2) - (a_4*c2)/(1 + b_4*p1 + b_6*p2) - (a_5*x)/(1 + b_5*p1 + b_5*p2 + b_3*c1 + b_3*c2))

	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c1_dot):
		c1_dot = np.nextafter(0,1)
	if math.isnan(c2_dot):
		c2_dot = np.nextafter(0,1)
	if math.isnan(p1_dot):
		p1_dot = np.nextafter(0,1)
	if math.isnan(p2_dot):
		p2_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c1_new = c1 + c1_dot*dt
	c2_new = c2 + c2_dot*dt
	p1_new = p1 + p1_dot*dt
	p2_new = p2 + p2_dot*dt

	return x_new, c1_new, c2_new, p1_new, p2_new


# Food web with three trophic levels and a semi-omnivorous predator
#
#	--------X
#	|	   / \
#	|	  /   \
#	|	 /     \	
#	|  C1       C2
#	|  |\	    /|
#	|  | \     / |
#   |  |  \   /  |
#	|  |   \ /	 |
#   |  |   / \   |
#   |  |  /   \  |
#   |  | /     \ |
#	|--P1       P2
#
def tritrophicPartialOmnivory(x, c1, c2, p1, p2, d_x, d_c1, d_c2, dt):

	x_dot = x*((a_3*c1 + a_3*c2 + a_5*p1)/(1 + b_3*c1 + b_3*c2 + b_5*p1) - d_x)
	c1_dot = c1*((a_4*p1 + a_6*p2)/(1+b_4*p1 + b_6*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2 + b_5*p1) - d_c1)
	c2_dot = c2*((a_6*p1 + a_2*p2)/(1+b_6*p1 + b_4*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2 + b_5*p1) - d_c2)
	p1_dot = p1*(r*(1-p1) - ef.f(c1,p1,a_6,b_6) - ef.f(c2,p1,a_6,b_6) - (a_5*x)/(1 + b_5*p1 + b_3*p1))
	p2_dot = p2*(r*(1-p2) - ef.f(c1,p2,a_6,b_6) - ef.f(c2,p2,a_6,b_6) - (a_5*x)/(1 + b_5*p1 + b_3*p1))

	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c1_dot):
		c1_dot = np.nextafter(0,1)
	if math.isnan(c2_dot):
		c2_dot = np.nextafter(0,1)
	if math.isnan(p1_dot):
		p1_dot = np.nextafter(0,1)
	if math.isnan(p2_dot):
		p2_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c1_new = c1 + c1_dot*dt
	c2_new = c2 + c2_dot*dt
	p1_new = p1 + p1_dot*dt
	p2_new = p2 + p2_dot*dt

	return x_new, c1_new, c2_new, p1_new, p2_new

# Food web with three trophic levels with each trophic level only feeding on
# the nearest lower trophic level
#
#	        X
#		   / \
#	 	  /   \
#	 	 /     \	
#	   C1       C2
#	   |\	    /|
#	   | \     / |
#      |  \   /  |
#	   |   \ /	 |
#      |   / \   |
#      |  /   \  |
#      | /     \ |
#	   P1       P2
#
def tritrophicNoOmnivory(x, c1, c2, p1, p2, d_x, d_c1, d_c2, dt):

	x_dot = x*((a_3*c1 + a_3*c2 + a_5*p1)/(1 + b_3*c1 + b_3*c2 + b_5*p1) - d_x)
	c1_dot = c1*((a_4*p1 + a_6*p2)/(1+b_4*p1 + b_6*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2) - d_c1)
	c2_dot = c2*((a_6*p1 + a_2*p2)/(1+b_6*p1 + b_4*p2) - (a_3*x)/(1 + b_3*c1 + b_3*c2) - d_c2)
	p1_dot = p1*(r*(1-p1) - ef.f(c1,p1,a_4,b_4) - ef.f(c2,p1,a_6,b_6))
	p2_dot = p2*(r*(1-p2) - ef.f(c1,p2,a_6,b_6) - ef.f(c2,p2,a_4,b_4))

	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c1_dot):
		c1_dot = np.nextafter(0,1)
	if math.isnan(c2_dot):
		c2_dot = np.nextafter(0,1)
	if math.isnan(p1_dot):
		p1_dot = np.nextafter(0,1)
	if math.isnan(p2_dot):
		p2_dot = np.nextafter(0,1)

	x_new = x + x_dot*dt
	c1_new = c1 + c1_dot*dt
	c2_new = c2 + c2_dot*dt
	p1_new = p1 + p1_dot*dt
	p2_new = p2 + p2_dot*dt

	return x_new, c1_new, c2_new, p1_new, p2_new


# Food web with 5 trophic levels with predator X having omnivory on all trophic
# levels below it
#
#  			Z
#			|			
#			Y 			
#			| 			
#	|-------X--------|
#	|	   /|\ 		 |
#	| 	  / | \		 |
#	| 	 /  |  \	 |
#	|  C1   |   C2 	 |
#	|  |\	|   /| 	 |
#	|  | \  |  / |	 |
#   |  |  \ | /  | 	 |
#	|  |   \|/	 |	 |
#   |  |   /|\   |	 |
#   |  |  / | \  |   |
#   |  | /  |  \ |	 |
#	|--P1   P2   P3--|
#
def quintrophicOmnivory(z, y, x, c1, c2, p1, p2, p3, d_z, d_y, d_x, d_c1, d_c2, dt):

	z_dot = z*(ef.f(y,y,a_1,b_1) - d_z)
	y_dot = y*(ef.f(x,x,a_2,b_2) - ef.f(z,y,a_1,b_1) - d_y)
	x_dot = x*( (a_5*p1 + a_5*p2 + a_5*p3 + a_3*c1 + a_3*c2)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2) - ef.f(y,x,a_2,b_2) - d_x)
	c1_dot = c1*((a_4*p1 + a_7*p2 + a_6*p3)/(1 + b_4*p1 + b_7*p2 + b_6*p3) - (a_3*x)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2) - d_c1)
	c2_dot = c2*((a_4*p1 + a_7*p2 + a_6*p3)/(1 + b_4*p1 + b_7*p2 + b_6*p3) - (a_3*x)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2) - d_c2)
	p1_dot = p1*(r*(1-p1) - (a_4*c1)/(1 + b_4*p1 + b_7*p2 + b_6*p3) - (a_6*c2)/(1 + b_6*p1 + b_7*p2 + b_4*p3) - (a_5*x)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2))
	p2_dot = p2*(r*(1-p2) - (a_7*c1)/(1 + b_4*p1 + b_7*p2 + b_6*p3) - (a_7*c2)/(1 + b_6*p1 + b_7*p2 + b_4*p3) - (a_5*x)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2))
	p3_dot = p3*(r*(1-p3) - (a_6*c1)/(1 + b_4*p1 + b_7*p2 + b_6*p3) - (a_4*c2)/(1 + b_6*p1 + b_7*p2 + b_4*p3) - (a_5*x)/(1 + b_5*p1 + b_5*p2 + b_5*p3 + b_3*c1 + b_3*c2))

	if math.isnan(z_dot):
		z_dot = np.nextafter(0,1)
	if math.isnan(y_dot):
		y_dot = np.nextafter(0,1)
	if math.isnan(x_dot):
		x_dot = np.nextafter(0,1)
	if math.isnan(c1_dot):
		c1_dot = np.nextafter(0,1)
	if math.isnan(c2_dot):
		c2_dot = np.nextafter(0,1)
	if math.isnan(p1_dot):
		p1_dot = np.nextafter(0,1)
	if math.isnan(p2_dot):
		p2_dot = np.nextafter(0,1)
	if math.isnan(p3_dot):
		p3_dot = np.nextafter(0,1)

	z_new = z + z_dot*dt
	y_new = y + y_dot*dt
	x_new = x + x_dot*dt
	c1_new = c1 + c1_dot*dt
	c2_new = c2 + c2_dot*dt
	p1_new = p1 + p1_dot*dt
	p2_new = p2 + p2_dot*dt
	p3_new = p3 + p3_dot*dt

	return z_new, y_new, x_new, c1_new, c2_new, p1_new, p2_new, p3_new


#_____________________________________________________________________________________________________________________________
#_____________________________________________________________FIN_____________________________________________________________
#_____________________________________________________________________________________________________________________________