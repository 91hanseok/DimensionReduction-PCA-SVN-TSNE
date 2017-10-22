import numpy as np 
from scipy import linalg as LA
import matplotlib.pyplot as plt
import sys

print("Read file:",sys.argv[1])
file=open(sys.argv[1],'r')
#file=open("pca_c.txt",'r')
raw_data=[]
for line in file:
	token=line.split('	')
	raw_data.append(token)
data=[]
for i in range(len(raw_data)):
	temp=[]
	for j in range(len(raw_data[i])-1):
		temp.append(float(raw_data[i][j]))
	data.append(temp)


hash_color=dict()
color_idx=0
for i in range(len(raw_data)):
	if raw_data[i][-1] not in hash_color:
		hash_color[raw_data[i][-1]]=color_idx
		color_idx+=1

data_catagory=[]
for i in range(len(raw_data)):
	data_catagory.append(hash_color[raw_data[i][-1]])
data=np.transpose(np.array(data))
for i in range(len(data)):
		mean=np.mean(data[i])
		for j in range(len(data[i])):
			data[i][j]-=mean

U, s, V = LA.svd( np.transpose(data),full_matrices=False )

s[2:]=0

new_U=np.dot(U[:,0:2],np.identity(2)*s[:2])
print("Reduced data:",new_U)
#new_U
data_catagory=dict()
for i in range(len(raw_data)):
	if raw_data[i][-1] not in data_catagory:
		data_catagory[raw_data[i][-1]]=np.array(new_U[i,:])
	else :
		data_catagory[raw_data[i][-1]]=np.vstack((data_catagory[raw_data[i][-1]],new_U[i,:]))

for key, value in data_catagory.iteritems():

	plt.scatter(value[:,0],value[:,1],s=30,label=key[:-2])
#color = [str(item/255.) for item in data_catagory]
#plt.scatter(new_U[:,0:1], new_U[:,1:2], s=30)
plt.title("SVD: "+str(file.name))
plt.legend()
#print(new_U[:,0:1])
#
plt.show()


