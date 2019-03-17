
# coding: utf-8

# In[13]:

from scipy.stats import norm, t
def power_in_hypo_test_for_mean(mu_b, mu_h, n, alpha, sd, pop = True, tail = -1):
    assert n > 0
    assert sd > 0
    assert alpha > 0 and alpha < 1
    try:
        if tail not in (-1, 0, 1):
            raise ValueError
    except ValueError:
        print ('Test type indicator not found')
    else:
        if pop:
            if tail == -1:
                if mu_b < mu_h:
                    z = norm.ppf(alpha)
                    m = z*sd/n**0.5 + mu_h # critical population mean
                    score = (m - mu_b)/(sd/n**0.5)
                    p = norm.cdf(score) #probability of rejecting H0 when it is false
                    return (p)
                else:
                    return print ('Cannot find the power because H0 is true')
            elif tail == 1:
                if mu_b > mu_h:
                    z = norm.ppf(1-alpha)
                    m = z*sd/n**0.5 + mu_h # critical population mean
                    score = (m - mu_b)/(sd/n**0.5)
                    p = 1 - norm.cdf(score) #probability of rejecting H0 when it is false
                    return (p)
                else:
                    return print('Cannot find the power because H0 is true')
            else:
                if mu_b != mu_h:
                # power for lower tail
                    z1 = norm.ppf(alpha/2)
                    m1 = z1*sd/n**0.5 + mu_h # critical population mean
                    score1 = (m1 - mu_b)/(sd/n**0.5)
                    p1 = norm.cdf(score1) # probability of rejecting H0 when it is false with lower tail
                # power for upper tail
                    z2 = norm.ppf(1-alpha/2)
                    m2 = z2*sd/n**0.5 + mu_h # critical population mean
                    score2 = (m2 - mu_b)/(sd/n**0.5)
                    p2 = 1 - norm.cdf(score2) # probability of rejecting H0 when it is false with upper tail
                # calculate power
                    p = p1 + p2 # probability of rejecting H0 when it is false
                    return (p)
                else:
                    return print('Cannot find the power because H0 is true')
        else:
            if tail == -1:
                if mu_b < mu_h:
                    t_score = t.ppf(alpha, n-1)
                    m = t_score*sd/n**0.5 + mu_h # critical sample mean
                    score = (m - mu_b)/(sd/n**0.5)
                    p = t.cdf(score, n-1) # probability of rejecting H0 when it is false
                    return (p)
                else:
                    return print('Cannot find the power because H0 is true')
            elif tail == 1:
                if mu_b > mu_h:
                    t_score = t.ppf(1-alpha, n-1)
                    m = t_score*sd/n**0.5 + mu_h # critical sample mean
                    score = (m - mu_b)/(sd/n**0.5) 
                    p = 1 - t.cdf(score, n-1) # probability of rejecting H0 when it is false
                    return print (p)
                else:
                    return print('Cannot find the power because H0 is true')
            else:
                if mu_b != mu_h:
                # test for lower tail
                    t1 = t.ppf(alpha/2, n-1)
                    m1 = t1*sd/n**0.5 + mu_h # critical sample mean
                    score1 = (m1 - mu_b)/(sd/n**0.5)
                    p1 = t.cdf(score1, n-1) # probability of rejecting H0 when it is false with lower tail
                # test for upper tail
                    t2 = t.ppf(1-alpha/2, n-1) 
                    m2 = t2*sd/n**0.5 + mu_h # critical sample mean
                    score2 = (m2 - mu_b)/(sd/n**0.5)
                    p2 = 1 - t.cdf(score2, n-1) # probability of rejecting H0 when it is false with upper tail
                # caluculate power
                    p = p1 + p2 # probability of rejecting H0 when it is false
                    return print (p)
                else:
                    return print('Cannot find the power because H0 is true')


# In[14]:

power_in_hypo_test_for_mean(23, 25, 30, 0.02, 3, True, -1)
power_in_hypo_test_for_mean(25, 25, 30, 0.02, 3, True, -1)
power_in_hypo_test_for_mean(23, 25, 30, 0.02, 3, True, 2)


# In[15]:

power_in_hypo_test_for_mean(17, 15, 35, 0.01, 4, True, 1)
power_in_hypo_test_for_mean(15, 15, 35, 0.01, 4, True, 1)


# In[16]:

power_in_hypo_test_for_mean(76, 75, 16, 0.05, 8, True, 0)
power_in_hypo_test_for_mean(77, 75, 16, 0.05, 8, True, 0)
power_in_hypo_test_for_mean(75, 75, 16, 0.05, 8, True, 0)


# In[17]:

power_in_hypo_test_for_mean(9950, 10000, 30, 0.05, 125, False, -1)
power_in_hypo_test_for_mean(2.09, 2.0, 35, 0.05, 0.3, False, 1)
power_in_hypo_test_for_mean(15.1, 15.4, 35, 0.05, 2.5, False, 0)


# In[18]:

power_in_hypo_test_for_mean(119.999, 120, 36, 0.05, 12, pop=False, tail=-1)
power_in_hypo_test_for_mean(120.001, 120, 36, 0.05, 12, pop=False, tail=1)
power_in_hypo_test_for_mean(16.5, 16, 30, 0.05, 0.8, pop=False, tail=0)


# In[19]:

power_in_hypo_test_for_mean(119.999, 120, 36, 0.05, 12, pop=True, tail=4)
power_in_hypo_test_for_mean(119.999, 120, 36, 0.05, 12, pop=False, tail=4)


# In[20]:

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

# plot power curve for lower-tailed test with population sd
x = np.arange(110, 120, 0.02)
plt.plot(x, power_in_hypo_test_for_mean(x, 120, 36, 0.05, 12, pop=True, tail=-1), 'r--')
plt.show()

# plot power curve for upper-tailed test with population sd
x = np.arange(120, 130, 0.02)
plt.plot(x, power_in_hypo_test_for_mean(x, 120, 36, 0.05, 12, pop=True, tail=1), 'r--')
plt.show()

# plot power curve for two-tailed test with population sd
x = np.arange(15, 17, 0.02)
plt.plot(x, power_in_hypo_test_for_mean(x, 16, 30, 0.05, 0.8, pop=True, tail=0), 'r--')
plt.show()


# In[21]:

list_x =[x for x in np.arange(110, 120, 0.02)]
list_y=[]
for x in np.arange(110, 120, 0.02):
    y = power_in_hypo_test_for_mean(x, 120, 36, 0.05, 12, pop=True, tail=-1)
    list_y.append(y)
plt.plot(list_x,list_y)
plt.show()


# In[8]:

from scipy.stats import norm, t
import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[12]:

for x in np.arange(110, 120, 0.02):
    plt.plot(x, power_in_hypo_test_for_mean(x, 120, 36, 0.05, 12, pop=True, tail=-1), 'r--')
    plt.show()


# In[ ]:



