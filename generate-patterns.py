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
		x=(np.random.rand(L)>(1*x0+0.0)).astype(int)
		if len(data):
			data=np.vstack((data,x))
		else:
			data=x
	return data
	
if __name__ == '__main__':
	n=100
	TA=3
	TB=7
	L=30
	dataA=generate_sinusoidal(n,L,TA)
	dataB=generate_sinusoidal(n,L,TB)
	dataC=generate_random(n,L)
	data=np.vstack((dataA,dataB,dataC))
	np.save('movement-patterns',data)
		
