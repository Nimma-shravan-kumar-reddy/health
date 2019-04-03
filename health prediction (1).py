
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("C:\\Users\\Admin\\Desktop\\health_set.csv")
a=df[:450]
df.head()


# In[4]:


# now let us see Gender vs Heart_rate
x=a["Gender"]
y=a["Heart_Rate"]
plt.bar(x,y)
plt.xlabel('Gender')
plt.ylabel('Heart_Rate')
plt.title('Gender vs Heart_Rate')
plt.show()


# In[5]:


# now let us see how heart_rate is depended on age
x=a["Age"]
y=a["Heart_Rate"]
plt.bar(x,y)
plt.xlabel('Age')
plt.ylabel('Heart_Rate')
plt.title('Age vs Heart_Rate')
plt.show()


# In[6]:


# let us see Height vs Heart_rate
fig = plt.figure(figsize = (25,6))
sns.barplot(x="Height",y="Heart_Rate",data=a)


# In[7]:


#Now let us find Weight vs Heart_Rate
fig = plt.figure(figsize = (25,8))
sns.barplot(x="Weight",y="Heart_Rate",data=a)


# In[8]:


fig = plt.figure(figsize = (25,8))
sns.barplot(x="Calories",y="Heart_Rate",data=a[:100])


# In[9]:


x=a["Calories"]
y=a["Heart_Rate"]
x=x[:15]
y=y[:15]
plt.scatter(x,y)


# In[10]:


x=a["Weight"]
y=a["Heart_Rate"]
x=x[:15]
y=y[:15]
plt.scatter(x,y)


# In[17]:


def estimate_coef(x,y):
    n=np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
    return(b_0, b_1) 
def plot_regression_line(x, y, b):
    plt.scatter(x, y, color = "m",marker = "o", s = 30)
    y_pred = b[0] + b[1]*x 
    plt.plot(x, y_pred, color = "g")
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.show() 
def main():
    x=a["Calories"]
    y=a["Heart_Rate"]
    x=x[:15]
    y=y[:15]
    b = estimate_coef(x, y) 
    print("Estimated coefficients:\nb_0 = {} b_1 = {}".format(b[0], b[1])) 
    plot_regression_line(x, y, b) 
if __name__ == "__main__": 
    main() 
b_0=82.3632391568
b_1=0.11859747236
    
        
  


# In[20]:


# 1. TEST CASES Input=66
# Actual output =  90.190
# Expected output = 94

x_input=66
y_prediction=b_0+(b_1)*x_input
print(y_prediction)


# In[21]:


# 2. TEST CASES Input=231
# Actual output =  109.759
# Expected output = 105
x_input=231
y_prediction=b_0+(b_1)*x_input
print(y_prediction)


# In[23]:


# 3. TEST CASES Input=26
# Actual output =  85.444
# Expected output = 88
x_input=26
y_prediction=b_0+(b_1)*x_input
print(y_prediction)


# In[25]:


# 4. TEST CASES Input=71
# Actual output =  90.7836
# Expected output = 100
x_input=71
y_prediction=b_0+(b_1)*x_input
print(y_prediction)


# In[28]:


# 5. TEST CASES Input=35
# Actual output =  86.51415
# Expected output = 81
x_input=35
y_prediction=b_0+(b_1)*x_input
print(y_prediction)

