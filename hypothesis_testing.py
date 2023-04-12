#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For each of the following questions, formulate a null and alternative hypothesis (be as specific as you can be), 
# then give an example of what a true positive, true negative, type I and type II errors would look like. Note 
# that some of the questions are intentionally phrased in a vague way. It is your job to reword these as more 
# precise questions that could be tested.


# In[ ]:


#     Has the network latency gone up since we switched internet service providers?

# H0 = The network latency has not changed since switching ISPs
# HA = The network latency has increased since switching ISPs (i.e. the internet is slower)

# True positive: We reject H0 in favor of HA AND HA is true 
# -- We conclude internet has gotten slower AND it has actually gotten slower

# True negative: We fail to reject the H0 AND H0 is true 
# -- We conclude the internet is the same speed AND it is actually the same speed

# Type I error is a false positive: H0 is true, but we reject it in favor of HA 
# -- we conclude the internet is slower, but actually it is not slower

# Type II error is a false negative: We fail to reject H0, but actually HA is true 
# -- We conclude internet speed has not changed, BUT actually it has changed


# In[ ]:


#     Is the website redesign any good?

# H0 = website redesign has had no change in usefulness. We'll measure in terms of number of page visits 
# HA = website redesign has increasead in usefulness (num of page visits has increased) -- (Do we need to declare an amount for the increase?)

# True Pos: We reject H0 in favor of HA AND HA is true (number of page visits has increased)

# True Neg: We fail to reject H0 AND H0 is true (number of page visits hasn't changed)

# Type I Error (False Positive): We reject H0 in favor of HA, but H0 is true
#  -- We falsely conclude the number of page visits has increased

# Type II Error (False Negative): We fail to reject H0 when HA is true
#  -- We falsely conclude there is not an increase in page visits, when actually there is an increase


# In[ ]:


#     Is our television ad driving more sales?

# H0 = the tv ad has had no impact on sales
# HA = the tv ad has increased sales


# True Pos: We reject H0 in favor of HA AND HA is true
# -- We conclude the ad has increased sales AND the ad has increased sales
# -- ACTUALLY: can we determine causality?

# True Neg: We fail to reject H0 AND H0 is true
# -- We conclude sales have not increased AND they actually have not increased

# Type I Error (False Positive): We reject H0 in favor of HA, but H0 is true
# -- We conclude sales have increased, BUT actually they have not increased

# Type II Error (False Negative): We fail to reject H0 when HA is true
# -- We conclude sales have not increased BUT actually they have increased


# In[2]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)


# In[37]:


# Answer with the type of test you would use (assume normal distribution):

#     Is there a difference in grades of students on the second floor compared to grades of all students?
# Answer: One-Sample, two-tail t-test because we comparing a group that is within the larger group

#     Are adults who drink milk taller than adults who dont drink milk?
# Answer: Independent one-tail t-test (aka 2-sample, one-tail t-test) because we have two groups 
#         that do not overlap with each other

#     Is the the price of gas higher in texas or in new mexico?
# Answer: Independent, one-tail t-test (aka 2-sample, one-tail t-test) because we have two groups 
#         that do not overlap with each other

#     Are there differences in stress levels between students who take data science vs students who take 
# web development vs students who take cloud academy?
# Answer: Analysis of Variance aka ANOVA because we have more than 2 groups and we are trying to determine
#         if there are differences between the groups. By definition, two-tail, since ANOVA can't test for >, <


# In[ ]:


# Ace Realty wants to determine whether the average time it takes to sell homes is different for its two offices. 
# A sample of 40 sales from office #1 revealed a mean of 90 days and a standard deviation of 15 days. A sample of 
# 50 sales from office #2 revealed a mean of 100 days and a standard deviation of 20 days. Use a .05 level of 
# significance.


# In[39]:


# two-sample, two-tail test

# Step 1: Plot distributions via histogram
#  First make the distributions

office1_days_dist = stats.norm(90, 15)
office2_days_dist = stats.norm(100, 20)


# In[45]:


office1_sim = office1_days_dist.rvs(40)
plt.hist(office1_sim)
plt.show()
office1_sim.std()


# In[46]:


office2_sim = office2_days_dist.rvs(50)
plt.hist(office2_sim)
plt.show()


# In[ ]:


# Step 2: Establish Hypothesis:
# H0: There is NO difference in the average number of days it takes to sell a home between office 1 and office 2
# Ha: There IS a difference (so this is a 2 tailed test; later we will test for p<alpha)


# In[47]:


# Step 3: Set alpha = .05
alpha = .05


# In[48]:


# Step 4: Verify assumptions:
# Independence - Yes because the data points / distributions come from two different groups
# Normal distributioin? - Yes because the question said so, and I created normal dist's via stats.norm
# Equal variances? use stats.levene
s, p = stats.levene(office1_sim, office2_sim)
s, p
# p is < alpha so variances are NOT equal. So we set equal_var = False in the next step
# Also
# print(office1_sim.var())
# print(office2_sim.var())


# In[49]:


# Step 5: Compute test statistic and probability (t-statistic & p-value) using stats.ttest_ind
t, p = stats.ttest_ind(office1_sim, office2_sim, equal_var=False)
t, p


# In[50]:


if p < alpha:
    print("There IS a difference between the average number of days it takes to sell a house between office 1 and 2.")
else:
    print("There is NO difference between the avg num of days it takes to sell a house btwn office 1 and 2.")


# In[21]:


# Load the mpg dataset and use it to answer the following questions:

from pydataset import data
df = data('mpg')
df.head()


# In[52]:


#     Is there a difference in fuel-efficiency in cars from 2008 vs 1999?

# Since we're comparing two different groups, I will use the stats.ttest_ind, aka 2-sample, 2-tail t-test
# Step 1: Plot distributions via histogram
# First capture data into two arrays (_1999 and _2008) where each element is the average of cty and hwy
# for all cars with model years 1999 and 2008, respectively.

fuel_eff_1999 = (df[df.year==1999].cty + df[df.year==1999].hwy) / 2
fuel_eff_2008 = (df[df.year==2008].cty + df[df.year==2008].hwy) / 2

plt.figure(figsize=(9,6))
fuel_eff_1999.hist()
fuel_eff_2008.hist()
# sns.histplot(fuel_eff_1999)
# sns.histplot(fuel_eff_2008)
# plt.show()
# fuel_eff_1999.mean(), fuel_eff_2008.mean()


# In[32]:


# step 2: Establish hypothesis:
# H0: There is NO difference between fuel efficiency of cars from 1999 and cars from 2008
# Ha: There IS a difference (two-tail test, so just testing for p<alpha)

# step 3: Set alpha = .05
alpha = .05


# In[34]:


# step 4: verify assumptions (ind, normal, equal variance)
# indepence? - Yes. one group's data points do not depend on the other group(?)
# normal? - Yes because we have > 30 data points in each group (117 in both grps to be precise)
# variance? - Use stats.levene
s, p = stats.levene(fuel_eff_1999, fuel_eff_2008)
s, p
# p is .8555... which is > alpha of .05, so the two distributions have "equal" variances


# In[54]:


# step 5: Compute t-statistic and p value
t, p = stats.ttest_ind(fuel_eff_1999, fuel_eff_2008)
t, p
# p is .82637... which is > alpha, so we fail to reject H0


# In[55]:


# step 6, make a decision
# p is .82637... which is > alpha, so we fail to reject H0

if p < alpha:
    print("There IS a difference between the fuel efficiency of cars made in 1999 and 2008.")
else:
    print("There is NO significant difference between the fuel efficiency of cars made in 1999 and 2008.")


# In[50]:


####     Are compact cars more fuel-efficient than the average car?

# Since we're comparing one group that is within the larger group I'll use stats.ttest_1samp 
# aka One-sample t-test, AND it will be a 1 tail test
# Step 1: Plot distributions via histograms:

fuel_eff_compact = (df[df['class']=='compact'].cty + df[df['class']=='compact'].hwy) / 2
fuel_eff_overall = (df.cty + df.hwy) / 2

plt.hist(fuel_eff_overall)
plt.show()


# In[51]:


plt.hist(fuel_eff_compact)
plt.show()


# In[52]:


# Step 2: Establish Hypothesis
# H0 - Fuel eff of compact cars is <= fuel eff of all cars
# Ha - Fuel eff. of compact cars is > fuel eff of all cars

# Step 3: Set significance level, alpha = .05
alpha = .05


# In[53]:


# Step 4: Verify assumptions: normal or >= 30 observations
fuel_eff_compact.shape
# We have 47 observations for compact and way more than 47 for all cars, so YES we pass the normal assumption


# In[54]:


# Step 5: Compute test statistics
t, p = stats.ttest_1samp(fuel_eff_compact, fuel_eff_overall.mean())
t, p


# In[74]:


# step 6, make a decision
# p is 4.198...e-10 which is < alpha, so we reject H0

if (p/2 < alpha) and (t > 0):
    print("The fuel efficiency of compact cars is greater than the fuel eff. of all cars.")
else:
    print("The fuel efficiency of compact cars is less than or equal to the fuel eff. of all cars.")


# In[57]:


#     Do manual cars get better gas mileage than automatic cars?

# I'll use a Independent t-test aka a 2-sample, 1-tail t-test since we have two distinct groups
# Step 1, plot distributions via histograms

fuel_eff_man = (df[df.trans.str.contains('man')].cty + df[df.trans.str.contains('man')].hwy) / 2
fuel_eff_auto = (df[df.trans.str.contains('auto')].cty + df[df.trans.str.contains('auto')].hwy) / 2

plt.hist(fuel_eff_man)
plt.show()


# In[58]:


plt.hist(fuel_eff_auto)
plt.show()


# In[59]:


# Step 2: Establish hypothesis
# H0: Fuel eff of manuals is <= automatics
# Ha: Fuel eff of manuals is > automatics

# Step 3: Set alpha = .05
alpha = .05


# In[60]:


# Step 4: Verify assumptions: Independence, normal dist, equal variances
# Independence? Yes, each data point does not depend on any other data point
# Normal: We have >= 30 data points (77 for man, 157 for auto)
fuel_eff_man.shape, fuel_eff_auto.shape


# In[61]:


# Equal Variance? - use stats.levene
s, p = stats.levene(fuel_eff_man, fuel_eff_auto)
s, p
# p is .6545... which is > alpha of .05, so Equal Variance


# In[62]:


# Step 5: Compute statistics via ttest_ind
t, p = stats.ttest_ind(fuel_eff_man, fuel_eff_auto)
t, p


# In[73]:


# Step 6: Make Decision
# since p/2 is < alpha and t is > 0, we can reject H0 in favor of Ha

if (p/2 < alpha) and (t > 0):
    print("Manuals have greater fuel efficiency than automatics.")
else:
    print("Manuals do NOT have greater fuel efficiency than automatics")


# In[2]:


############ Correlation Exercises #################


# In[ ]:


#     Answer with the type of stats test you would use (assume normal distribution):

#         Is there a relationship between the length of your arm and the length of your foot?
# Answer: 2 x continuous variables; testing for relationship:  stats.pearsonr

#         Do guys and gals quit their jobs at the same rate?
# Answer: 2 groups and we want to compare means:  2-sample, probably 2-tail t-test: stats.ttest_ind

#         Does the length of time of the lecture correlate with a students grade?
# Answer: 2 x continuous variables; testing for linear relationship: stats.pearsonr


# In[18]:


#     Use the telco_churn data.
#         Does tenure correlate with monthly charges?
#         Total charges?
#         What happens if you control for phone and internet service?

from env import host, user, password
def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

connection_str = get_db_url('telco_churn')
query = """
            SELECT * 
            FROM customers
            JOIN customer_churn USING (customer_id)
            JOIN internet_service_types USING (internet_service_type_id)
        """
df = pd.read_sql(query, connection_str)


# In[174]:



df.total_charges = df.total_charges.astype(float)
df.info()


# In[7]:


# Does tenure correlate with monthly charges?

# Step 1: What test?  2 x cont. var's; assume normal dist; testing for correlation: stats.pearsonr

# Step 2: Setup
# H0: There is NO linear relationship between tenure and monthly charges
# Ha: There IS a linear relationship between tenure and monthly charges
alpha = .05


# In[9]:


# Step 3: Visualize

sns.scatterplot(data = df, x='tenure', y = 'monthly_charges') 
plt.show()


# In[10]:


# Step 4: calculate statistic
r, p = stats.pearsonr(df.tenure, df.monthly_charges)
print ('r = ', r)
print ('p = ', p)


# In[175]:


# Step 5: Conclude
# Since p > alpha, we could normally reject the H0. However, the r value is < .5. 
# so ... the relationship isn't really strong enough to conclude that there is a relationship

# Turns out the data was NOT normally distributed, so I should have used spearmanr

r, p = stats.spearmanr(df.tenure, df.monthly_charges)
r, p


# In[ ]:




##### Does tenure correlate with total charges?  (makes sense that this would be true, but let's see)

# Step 0: look at data. I missed some blanks in the total charges column (should have used sort_values to see)

# Step 1: Which test:  same logic: stats.pearsonr
## Also missed the fact that the data was not normally distributed. CLT could account for it since we have
# tons of data points, but also could use spearmanr
# Step 2: Setup
# H0: There is NO linear relationship between tenure and total charges
# Ha: There IS a linear relationship between tenure and total charges
alpha = .05


# In[27]:


#Step 3: Visualize

sns.scatterplot(data = df, x='tenure', y = 'total_charges')
plt.show()


# In[30]:


# Step 4: calculate statistic
r, p = stats.pearsonr(df.tenure, df.total_charges)
print ('r = ', r)
print ('p = ', p)


# In[ ]:


# Step 5: Since r is near 1.0 and p is so low that python rounded it to zero (i.e. < alpha),
# We can reject the H0 which suggests Ha, that there is a correlation between tenure and total_charges


# In[45]:


# What happens if we control for phone and internet service
# So, I'll assume this means what if we control for phone and internet services in the first question
# which was asking about a correlation between tenure and monthly charges

# Steps: 1-stats.pearsonr, 2-H0 No correlation;Ha IS correlation
# Step 3: Visualize

sns.relplot(data = df, x='tenure', y = 'monthly_charges', hue='internet_service_type')
plt.show()


# In[48]:


# Step 4: compute the statistics with stats.pearsonr

# First, separate the df into three dfs that only include each internet service type, respectively

df.internet_service_type.value_counts()

fiber_df = df[df.internet_service_type=='Fiber optic']
dsl_df = df[df.internet_service_type=='DSL']
no_intrnt_df = df[df.internet_service_type=='None']

r1, p1 = stats.pearsonr(fiber_df.tenure, fiber_df.monthly_charges)
r2, p2 = stats.pearsonr(dsl_df.tenure, dsl_df.monthly_charges)
r3, p3 = stats.pearsonr(no_intrnt_df.tenure, no_intrnt_df.monthly_charges)

print ('Fiber Optic, r1 = ', r1,'  p1 = ', p1)
print ('DSL,         r2 = ', r2,'  p2 = ', p2)
print ('No Int Svc   r3 = ', r3,'  p3 = ', p3)


# In[ ]:


# Step 5: Conclude
# Well, each internet service type had positive r values, but only Fiber Optic had a positive r value > .5.
# All p values were well below .05
# So this suggests there is a linear relationship between tenure and monthly_charges for Fiber Optic Int Svc Type

# Just thinking out loud.  This may not tell us much of anything except that the company raises its monthly
# charges over time.  As one does.  It's not helping with churn (maybe I'll look at that later.)


# In[50]:


###### Next I'll control for phone type when comparing tenure and monthly charges #######

# Steps: 1-stats.pearsonr, 2-H0 No correlation;Ha IS correlation
# Step 3: Visualize

sns.relplot(data = df, x='tenure', y = 'monthly_charges', col='phone_service')
plt.show()


# In[51]:


# Step 4, compute statistic
# First separate df by phone service

hasphone_df = df[df.phone_service=='Yes']
nophone_df = df[df.phone_service=='No']

r1, p1 = stats.pearsonr(hasphone_df.tenure, hasphone_df.monthly_charges)
r1, p1 = stats.pearsonr(nophone_df.tenure, nophone_df.monthly_charges)

print ('Has phone service, r1 = ', r1,'  p1 = ', p1)
print ('No phone service,  r2 = ', r2,'  p2 = ', p2)


# In[ ]:


# Step 5, Conclude
# The r value for the hasphone_df is > .5 with a p1 < .05 (alpha), so we can reject the H0 that there is no 
# relationship between tenure and monthly charges in the group of customers that have phone service
# This suggests there IS a correlation between tenure and monthly_charges for customers w/ phone service


# In[52]:


###### New Question, new database #######

#     Use the employees database.
#         Is there a relationship between how long an employee has been with the company and their salary?
#         Is there a relationship between how long an employee has been with the company and the number of 
#             titles they have had?


# In[57]:


# # First up, comparing length of time with company vs. current salary
# This query takes several minutes, FYI

from env import host, user, password
connection_str = get_db_url('employees')
query = """
            SELECT emp_no, hire_date, salary, from_date, to_date
            FROM employees
            JOIN salaries USING (emp_no)
        """
emp_salary_df = pd.read_sql(query, connection_str)


# In[65]:


emp_salary_df.info()


# In[147]:


# The following lines of code create a new column named tenure
# tenure is the number of days you get when you hire_date from to_date
# First though, if to_date is 9999-01-01, I make it now (ex, 2023-04-11)

# group the dataframe by emp_no by the max of to_date. 
new_df = pd.DataFrame(emp_salary_df.groupby('emp_no').to_date.max())
new_df.head()


# In[148]:


# then merge it back with the sql pull to get the current salaries
new_df = new_df.merge(emp_salary_df, how = 'inner', on=['emp_no', 'to_date'])


# In[149]:


# i found datetime out there to help me test for 9999-01-01 and to add in today's date

import datetime as dt
future = new_df['to_date'].iloc[0]
now = dt.date(year = 2023, month = 4, day = 11)


# In[150]:


# make a new series that I will modify and put back into the new_df

new_to_date_series = new_df.to_date
new_new_td_series = pd.Series([now if x == future else x for x in new_to_date_series])


# In[153]:


# put the new series back into the data frame and set the tenure 
# column by subtracting hire_date from to_date
# NOW, I have two continuous variables to compare: tenure and current salary
new_df['to_date'] = new_new_td_series
new_df['tenure'] = new_df.to_date - new_df.hire_date
new_df.head()


# In[164]:


new_df['tenure'] = new_df['tenure'].dt.days


# In[166]:


new_df.info()


# In[ ]:


# Step 1: Test will be a stats.pearsonr
# Step 2: Setup
# H0 = There is NO correlation between salary and tenure
# Ha = There IS a correlation


# In[167]:


# Step 3 Visualize
sns.relplot(data = new_df, x='tenure', y = 'salary')
plt.show()


# In[168]:


# That was weird because there is a hole in the data between about 65000 and 82000 days of tenure.
# I'm going to press for now, but something probably isn't right
# Turns out, this data stops in the early 2000's. If I would have used the max(to_date) that wasn't 9999-01-01,
# it would have made more sense.

# Step 4 Compute statistic
r, p = stats.pearsonr(new_df.tenure, new_df.salary)
r, p


# In[ ]:


# Step 5 Conclude
# r is ~.32 which is < .5. Since p is very low (< .05), I can reject the H0 
# and say there is a weak positive correlation between tenure and salary


# In[ ]:


#         Is there a relationship between how long an employee has been with the company and the number of 
#             titles they have had?


# In[169]:


# I need a dataframe with emp_no and number of titles to add back into my new_df that has tenure

from env import host, user, password
connection_str = get_db_url('employees')
query = """
            SELECT emp_no, COUNT(title) as cnt_title
            FROM employees
            JOIN titles USING (emp_no)
            GROUP BY emp_no
        """
emp_title_df = pd.read_sql(query, connection_str)


# In[171]:


# merge emp_title_df back into new_df
new_df = new_df.merge(emp_title_df, how = 'inner', on=['emp_no'])
new_df.head()


# In[176]:


sns.relplot(data = new_df, x='tenure', y = 'cnt_title')
plt.show()
# Since cnt_title is a discrete variable, we should not use anything that requires continuous variables
# So we should use something like ANOVA / Kruskal Wallis
# do the levene test, turns out you don't have equal variance, so we are forced into Kruskal Wallis

# H0: The median salary is the same for the various number of titles
# Ha: The median salary is NOT equal

# ... more plots and kruskal wallis:  We reject the H0


# In[ ]:





# In[ ]:


###### New Question, new database #######


#     Use the sleepstudy data.
#         Is there a relationship between days and reaction time?


# In[177]:


from pydataset import data
df = data('sleepstudy')


# In[178]:


# H0 There is no relationship between days and reaction time
# Ha There IS a relationship


# In[180]:


react_df = df[['Reaction','Days']]
react_df.head()
react_df.info()


# In[181]:


sns.relplot(data = df, x='Days', y = 'Reaction')
plt.show()


# In[183]:


# Discrete (days) - so ANOVA?  Let's try spearman since the number of "categories", i.e days is so large, i.e. 10
# and the distributions are not normal

r, p = stats.spearmanr(df.Days, df.Reaction)
r, p


# In[ ]:


# Conclude: p is low and r is positive and > .5 => correlation yes


# In[ ]:


# Answer with the type of stats test you would use (assume normal distribution):

#     Do students get better test grades if they have a rubber duck on their desk?
# Answer: t-test, 2-sample, 1 tail

#     Does smoking affect when or not someone has lung cancer?
# Answer: chi_square because we're comparing two categorical variables

#     Is gender independent of a person’s blood type?
# Answer: chi_square

#     A farming company wants to know if a new fertilizer has improved crop yield or not
# Answer: t-test, 2-sample, 1 tail

#     Does the length of time of the lecture correlate with a students grade?
# Answer: pearsonr or spearmanr because we're comparing 2 continuous variables

#     Do people with dogs live in apartments more than people with cats?
# Answer: chi_square because we're comparing two categorical variables


# In[25]:


# Use the following contingency table to help answer the question of whether using a macbook and being a 
# codeup student are independent of each other.

# We can reject the H0 which suggests the Ha, i.e. that there is a relationship between using a macbook and being
# a Codeup student

# Step 1: Form Hypothesis
# H0 = There is no association between being a Codeup student and using a Macbook
# Ha = There IS an association
alpha = .05

observed = [[49, 20], [1, 30]]
observed = pd.DataFrame(observed)
observed


# In[26]:


# Step 3: compute statistic with stats.chi2_contingency

chi2, p, degf, expected = stats.chi2_contingency(observed)

# print 'Observed Values' followed by a new line
print('Observed Values\n')

# print the values from the 'observed' dataframe
print(observed.values)

# print --- and then a new line, 'Expected Values', followed by another new line
print('---\nExpected Values\n')

# print the expected values array
print(expected.astype(int))

# print a new line
print('---\n')

# print the chi2 value, formatted to a float with 4 digits. 
print(f'chi^2 = {chi2:.4f}') 

# print the p-value, formatted to a float with 4 digits. 
print(f'p     = {p:.4f}')


# In[ ]:


# Conclude
# since p < alpha, we can reject the H0


# In[ ]:


###############################
# Choose another 2 categorical variables from the mpg dataset and perform a chi2

# contingency table test with them. Be sure to state your null and alternative hypotheses.


# In[2]:


from pydataset import data
df = data('mpg')
df.head()


# In[12]:


# I will compare the cyl and class categorical variables

# Renamed the 'class' column to a non-reserved string
# df = df.rename(columns={'class': 'cls'})

# Step 1: Form hypothesis
# H0 = There is no association between the cyl and cls categories
# Ha = There IS an association btwn cyl, cls categories
alpha = .05
df.info()


# In[15]:


# Step 2: make contingency table
observed = pd.crosstab(df.cls, df.cyl)
observed


# In[17]:


# Step 3: compute statistic with stats.chi2_contingency

chi2, p, degf, expected = stats.chi2_contingency(observed)

# print 'Observed Values' followed by a new line
print('Observed Values\n')

# print the values from the 'observed' dataframe
print(observed.values)

# print --- and then a new line, 'Expected Values', followed by another new line
print('---\nExpected Values\n')

# print the expected values array
print(expected.astype(int))

# print a new line
print('---\n')

# print the chi2 value, formatted to a float with 4 digits. 
print(f'chi^2 = {chi2:.4f}') 

# print the p-value, formatted to a float with 4 digits. 
print(f'p     = {p:.4f}')


# In[ ]:


# Since p is < alpha, we can reject the H0


# In[3]:


# Use the data from the employees database to answer these questions:

#     Is an employee's gender independent of whether an employee works in sales or marketing? 
#       (only look at current employees)


from env import host, user, password
def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

connection_str = get_db_url('employees')
query = """
            SELECT de.emp_no, e.gender, de.dept_no, d.dept_name
            FROM dept_emp as de
            JOIN departments as d USING (dept_no)
            JOIN employees as e USING (emp_no)
            WHERE de.to_date > NOW()
            ORDER BY de.emp_no
        """
df = pd.read_sql(query, connection_str)


# In[7]:


# Step 1: Form hypothesis
# H0 = There is NO association between gender and whether or not they work in sales or marketing
# Ha = There IS an association

# Step 2: Make contingency table
# First make a new column and new df for sales or marketing only

new_df = df[(df.dept_name == 'Sales') | (df.dept_name == 'Marketing')]


# In[10]:


observed = pd.crosstab(new_df.gender, new_df.dept_name)
observed


# In[11]:


# Step 3: compute statistic with stats.chi2_contingency

chi2, p, degf, expected = stats.chi2_contingency(observed)

# print 'Observed Values' followed by a new line
print('Observed Values\n')

# print the values from the 'observed' dataframe
print(observed.values)

# print --- and then a new line, 'Expected Values', followed by another new line
print('---\nExpected Values\n')

# print the expected values array
print(expected.astype(int))

# print a new line
print('---\n')

# print the chi2 value, formatted to a float with 4 digits. 
print(f'chi^2 = {chi2:.4f}') 

# print the p-value, formatted to a float with 4 digits. 
print(f'p     = {p:.4f}')


# In[ ]:


# Conclude:
# Since p > alpha of .05, we cannot reject the H0 
# (which suggests there is not an association btwn gender and whether an employee works in Sales or Marketing)


# In[12]:


###### #     Is an employee's gender independent of whether or not they are or have been a manager?

connection_str = get_db_url('employees')
query = '''
        SELECT emp_no, gender, is_manager
        FROM employees
        LEFT JOIN (
            SELECT emp_no,
                IF (True, True, False) as is_manager
            FROM dept_manager
            ) as dm USING (emp_no)
        '''
mngr_df = pd.read_sql(query, connection_str)


# In[62]:



mngr_df.is_manager = mngr_df.is_manager.fillna(False)
mngr_df.is_manager = mngr_df.is_manager.astype(bool)
mngr_df.info()


# In[63]:


# Step 1, Form hypothesis
# H0 = There is no association between gender and is_manager
# Ha = There IS an association

# Step 2 make contingency table

observed = pd.crosstab(mngr_df.gender, mngr_df.is_manager)
observed


# In[64]:


# Step 3: compute statistic with stats.chi2_contingency

chi2, p, degf, expected = stats.chi2_contingency(observed)

# print 'Observed Values' followed by a new line
print('Observed Values\n')

# print the values from the 'observed' dataframe
print(observed.values)

# print --- and then a new line, 'Expected Values', followed by another new line
print('---\nExpected Values\n')

# print the expected values array
print(expected.astype(int))

# print a new line
print('---\n')

# print the chi2 value, formatted to a float with 4 digits. 
print(f'chi^2 = {chi2:.4f}') 

# print the p-value, formatted to a float with 4 digits. 
print(f'p     = {p:.4f}')


# In[ ]:


# Conclude
# Since p > .05, we fail to reject the H0

