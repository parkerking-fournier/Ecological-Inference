import math

# A type II saturating functional response
def f(u, v, a, b):
	f_u = ((a*u)/(1+b*v))
	if math.isnan(f_u):
		f_u = np.nextafter(0,1)
	return f_u