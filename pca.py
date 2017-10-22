import numpy as np 
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import sys

def pca(input_data):
	print("pca")
	for i in range(len(input_data)):
		mean=np.mean(input_data[i])
		for j in range(len(input_data[i])):
			input_data[i][j]-=mean
	#print(input_data)
	cov=np.cov(input_data)
	
	e_val,e_vec=np.linalg.eigh(cov)
	max_two=np.argsort(e_val)
	print("e_val",e_val)
	print("e_vec",e_vec)
	
	new_dim1=np.dot(e_vec[:,max_two[-1]],input_data)
	new_dim2=np.dot(e_vec[:,max_two[-2]],input_data)
	return np.vstack((new_dim1,new_dim2))



if __name__ == '__main__':

	print("Read file:",sys.argv[1])
	file=open(sys.argv[1],'r')

	raw_data=[]
	for line in file:
		token=line.split('	')
		raw_data.append(token)

	data=np.zeros((len(raw_data),len(raw_data[0])-1))
	#print(data)
	for i in range(len(raw_data)):
		for j in range(len(raw_data[i])-1):
			data[i][j]=float(raw_data[i][j])
	result=np.transpose(pca(np.transpose(data)))

	print("Reduced data:",result)
	hash_color=dict()
	color_idx=0
	for i in range(len(raw_data)):
		if raw_data[i][-1] not in hash_color:
			hash_color[raw_data[i][-1]]=color_idx
			color_idx+=1

	data_catagory=dict()
	for i in range(len(raw_data)):
		if raw_data[i][-1] not in data_catagory:
			data_catagory[raw_data[i][-1]]=np.array(result[i,:])
		else :
			data_catagory[raw_data[i][-1]]=np.vstack((data_catagory[raw_data[i][-1]],result[i,:]))



	for key, value in data_catagory.iteritems():
		plt.scatter(value[:,0],value[:,1],s=30,label=key[:-2])
	plt.title("PCA: "+str(file.name))
	plt.legend()
	plt.show()

