import numpy as np

def generate_random(n,L):
	data=np.array([],int)
	for i in range(n):
		x=(np.random.rand(L)>0.5).astype(int)
		if len(data):
			data=np.vstack((data,x))
		else:
			data=x
	return data
	
def generate_sinusoidal(n,L,T):
	data=np.array([],int)
	for i in range(n):
		x0=(np.sin(np.arange(L)*2*np.pi/T)+1)*0.5
		x=(np.random.rand(L)>(x0)).astype(int)
		if len(data):
			data=np.vstack((data,x))
		else:
			data=x
	return data
		
