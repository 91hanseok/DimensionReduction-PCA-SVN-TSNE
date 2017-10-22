from sklearn.manifold import TSNE
import numpy as np
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
color = [str(item/255.) for item in data_catagory]

model = TSNE(learning_rate=200, n_components=2, random_state=0, perplexity=60)

tsne = model.fit_transform(data)
print("Reduced data:",tsne)

data_catagory=dict()
for i in range(len(raw_data)):
	if raw_data[i][-1] not in data_catagory:
		data_catagory[raw_data[i][-1]]=np.array(tsne[i,:])
	else :
		data_catagory[raw_data[i][-1]]=np.vstack((data_catagory[raw_data[i][-1]],tsne[i,:]))

for key, value in data_catagory.iteritems():
	plt.scatter(value[:,0],value[:,1],s=30,label=key[:-2])

#print(tsne5)
plt.title("tsne: "+str(file.name))
plt.legend()
#plt.scatter(tsne[:, 0], tsne[:, 1],s=30, c=color)
plt.show()