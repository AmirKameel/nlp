#!/usr/bin/env python
# coding: utf-8

# # Import sklearn

# In[13]:


from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# # import model with good performance

# model = SentenceTransformer('all-MinilM-l6-v2')

# # Request is a libirary easliy to import data set

# In[14]:


import requests
response = requests.get('https://raw.githubusercontent.com/AmirKameel/nlp/main/data.txt')
data = response.text.split('\r\n')


# In[15]:


len(data)


# In[16]:


print(data)


# # converting the text data to matrix (vector) 

# In[17]:


vectorization= model.encode(data)


# In[18]:


print(vectorization)


# # Apply the K-MEANS

# In[19]:


num_clusters = 6
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(vectorization)
assin=clustering_model.labels_


# In[20]:


assin


# # Making nums of empty list of a list with num of clusters

# In[21]:


clusterd_sentence = [[] for i in range(num_clusters)]
clusterd_sentence


# # Assin the clusterd sentences to the empty lists 

# In[11]:


for sentence_id , cluster_id in enumerate(assin):
    clusterd_sentence[cluster_id].append(data[sentence_id])


# # Print the output and start with num 1

# In[12]:


for i , cluster in enumerate(clusterd_sentence):
    print("cluster" ,i+1)
    print(cluster)
    print()


# In[ ]:




