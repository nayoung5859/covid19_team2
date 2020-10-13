#!/usr/bin/env python
# coding: utf-8

# # [1] 표준화 프로세스

# In[2]:


import os 
from os.path import join
import copy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[48]:


df = pd.read_excel('19년 평균 합본_유동,직장,집객,상주,점포_최종.xlsx')
df.shape


# In[49]:


df = df.drop(columns = ['Unnamed: 0'])
df.head(1)


# In[8]:


df.describe()


# In[ ]:


# 표준화를 할 때, string값이 들어가있는 컬럼이 있으면 에러가 남
# 그래서 숫자로 되어있는 각 변수들만 가져옴!


# In[36]:


df1 = df[['유동인구', '직장인구', '집객시설', '상주인구', '점포수']]
df1.head(1)


# # Min-Max 스케일링

# In[20]:


from sklearn.preprocessing import MinMaxScaler

mMscaler = MinMaxScaler()


# #### 회귀분석을 할 때처럼 fit을 먼저해주고 transform을 해줘야함
# * fit_transform을 해주면 위 두 과정을 한번에 해줄 수 있다

# In[28]:


normal = mMscaler.fit_transform(df1) # transform부터가 0-1으로 바꾸는 것
normal = pd.DataFrame(normal, columns = df1.columns)
normal


# In[37]:


normal.describe()


# # 정규분포로 scaling하기

# In[38]:


from sklearn.preprocessing import StandardScaler
sdscaler = StandardScaler()


# In[39]:


sdscaler.fit(df1)


# In[40]:


normal2 = sdscaler.transform(df1)


# In[43]:


normal2 = pd.DataFrame(normal2, columns = df1.columns)
normal2


# In[53]:


normal2 = normal2.reset_index()
normal2.head(1)


# # 정규분포 스케일링+상권명 합치기

# In[50]:


df2 = df[['상권코드명']]
df2.head(1)


# In[54]:


df2 = df2.reset_index()
df2.head()


# In[56]:


normal_dist = pd.merge(df2, normal2, how = 'inner', on ='index')
normal_dist


# In[59]:


del normal_dist['index']


# In[60]:


del normal_dist['level_0']


# In[66]:


normal_dist.head(1)
normal_dist.shape


# In[62]:


normal_dist.to_excel('정규분포_표준화_유동,직장,집객,상주,점포.xlsx')


# # 클러스터링 시도!
# * K-means는 군집별 평균을 활용한다
# * 실루엣은 evaluation metrics

# In[67]:


normal_dist.shape


# In[69]:


normal_dist.describe()


# #### 5개 컬럼으로 이루어진 벡터는 우리가 눈으로 보기 어렵다. 따라서 2차원으로 변환해주는 과정 (=차원축소)이 필요한데 그게 바로 PCA. 
# * 문제는 여기서 상권명이 들어가있어서 자꾸 오류가 난다. 그래서 normal2인 표준화한 raw값만 들어있는 df을 활용해줬음

# In[75]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(normal2)


# In[76]:


data.shape


# #### k-means++이 뭔지 알아봐야함

# In[89]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)


# In[90]:


kmeans.fit(data) #학습완료


# In[108]:


cluster = kmeans.predict(data)# 각 클러스터 번호를 반환해줌
df_cl = pd.DataFrame(cluster, columns =['cluster'])
df_cl.reset_index()
df_cl.head()


# In[116]:


clustering= clustering.drop(columns = ['index'])


# In[120]:


clustering = pd.concat([df_cl, df2], axis = 1)
clustering.tail(1)
clustering = clustering.drop(columns= ['index'])


# In[125]:


clustering = pd.merge(clustering, df, how ='inner', on = '상권코드명')
clustering.head(1)


# In[131]:


clustering.to_excel('클러스터링 첫 시도.xlsx')


# #### 얘야 이게 무슨일이니..ㅠㅠ 왜 이렇게 나와 ㅠㅠ

# In[99]:


plt.scatter(data[:,0], data[:, 1], c=cluster, linewidth=1, edgecolor='black')
plt.show()


# In[ ]:




