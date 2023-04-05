#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Statistics Exercises


# In[1]:


import numpy as np
import pandas as pd

np.random.seed(42)


# In[25]:


# How likely is it that you roll doubles when rolling two dice?

# Initialize variables. outcomes is all the possible outcomes of rolling a die, trials, simulations
outcomes = [1,2,3,4,5,6]
n_trials = 2
n_simulations = 100_000

# make an array of 100,000 simulations of two dice rolls
two_dice_sim = np.random.choice(outcomes, (n_simulations, n_trials))

# use list comprehension to make a boolean array testing for equality of the two dice rolls
bool_tds = np.array(list((trial[0] == trial[1]) for trial in two_dice_sim))

# Another way:  two_dice_sim[:,0] == two_dice_sim[:,1]   -- pretty cool!
# Another way: make it a dataframe -> two_dice_sim[0] == two_dice_sim[1]

# finally, take the mean which is the chance of rolling two dice and having them be equal,
# i.e. 1-1, 2-2, 3-3, 4-4, 5-5, 6-6
bool_tds.mean()


# In[32]:


# If you flip 8 coins, what is the probability of getting exactly 3 heads? What is the probability of 
# getting more than 3 heads?

# same methodology as above
outcomes = ['H','T']
n_trials = 8
n_simulations = 100_000

coin_flips_8 = np.random.choice(outcomes, (n_simulations, n_trials))
bool_cf8 = coin_flips_8 == 'H'
num_heads_array = bool_cf8.sum(axis=1)
(num_heads_array == 3).mean()
(num_heads_array >= 3).mean()


# In[41]:


# There are approximitely 3 web development cohorts for every 1 data science cohort at Codeup. Assuming that 
# Codeup randomly selects an alumni to put on a billboard, what are the odds that the two billboards I drive 
# past both have data science students on them?

outcomes = ['WD', 'WD', 'WD', 'DS']
n_trials = 2
n_simulations = 100_000

billboard_combos = np.random.choice(outcomes, (n_simulations, n_trials))
bool_bc = (billboard_combos == 'DS')
(bool_bc.sum(axis=1) == 2).mean()


# In[63]:


# Codeup students buy, on average, 3 poptart packages with a standard deviation of 1.5 a day from the snack 
# vending machine. If on monday the machine is restocked with 17 poptart packages, how likely is it that I 
# will be able to buy some poptarts on Friday afternoon? (Remember, if you have mean and standard deviation, 
# use the np.random.normal) You'll need to make a judgement call on how to handle some of your values

# Initialize inputs to np.random.normal
mu = 3
sigma = 1.5
# n_trials is 5 for the 5 days in the week M-F
n_trials = 5
n_simulations = 100_000

# This line generates an array of 100_000 rows, each with 5 elements. Each element is an integer picked using
# the np.random.normal function which uses 3 for a mean and 1.5 for a std deviation. That returns a decimal, so I
# round it and make it an integer since codeup students can only ever buy whole numbers of poptart packages
poptart_sims = np.array(list((np.random.normal(mu, sigma, n_trials).round(0).astype(int) for i in range(n_simulations))))

# Finally, sum along the row axis and test if each row sum is >= 17 
# and finally finally take the mean (which is the decimal probability of running out of pop tarts)
(poptart_sims.sum(axis=1) >= 17).mean()


# In[ ]:





# In[ ]:




