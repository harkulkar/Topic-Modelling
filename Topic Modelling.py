#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


npr=pd.read_csv("npr.csv")


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[4]:


tfidf_instance=TfidfVectorizer(max_df=0.9,min_df=2,stop_words='english')


# In[10]:


dtm=tfidf_instance.fit_transform(npr['Article'])


# In[11]:


dtm


# In[12]:


#11992 articles ,54777 terms/words


# In[13]:


from sklearn.decomposition import NMF


# In[14]:


#LDA_instance=LatentDirichletAllocation(n_components=7,random_state=42)


# In[15]:


NMF_instance=NMF(n_components=7,random_state=42)


# In[ ]:





# In[16]:


NMF_instance.fit(dtm)


# In[18]:


tfidf_instance.get_feature_names_out()[2300]


# In[37]:


for i,topic in enumerate(NMF_instance.components_):
    print(f"The top 15 words for the topic#{i}")
    print([tfidf_instance.get_feature_names_out()[index] for index in topic.argsort()[-15:]])
    print()
    print()


# In[38]:


topic_results=NMF_instance.transform(dtm)


# In[39]:


topic_results[0]


# In[40]:


topic_results.argmax(axis=1)


# In[41]:


npr['Topic']=topic_results.argmax(axis=1)


# In[42]:


npr.head()


# In[43]:


my_topic_dict={0:'health',1:'politics',2:'insurance',3:'internation politics',4:'election',5:"music",6:'edu'}


# In[44]:


npr['Topic Label']=npr['Topic'].map(my_topic_dict)


# In[45]:


npr.head()


# In[ ]:





# In[47]:


# LDA_instance.components_


# In[48]:


dtm


# In[49]:


npr


# In[50]:


#we want to assign topic number to ever  doc


# In[53]:


# topic_results=LDA_instance.transform(dtm)


# In[54]:


topic_results.shape


# In[ ]:





# In[55]:


topic_results[0]


# In[56]:


topic_results[0].round(2)


# In[57]:


topic_results[0].argmax()


# In[58]:


npr["Topic"]=topic_results.argmax(axis=1)


# In[59]:


npr

