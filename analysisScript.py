#!/usr/bin/env python
# coding: utf-8

# # Analysis
# 
# In this notebook we will analyse the models that we created.
# 
# The analysis consists of:
# 
# 1) comparing the perplexity and the accuracy between the rationalized and non-rationalized model 
# 
# 2) Checking the change in perplexity when removing even more from the rational
# 
# 3) Checking the distribution of rationals
# 
# 4) Qualitative analysis of the some examples

# ## Setup

# In[1]:


#First we fix the relative imports
import os
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str)
args = parser.parse_args()

# In[2]:


### We load the models based on the configs for analysis. 
from utils.analysis import parse_config_for_analysis
config_path = args.config #'configs/dialoGPT_REINFORCE_dailyDialog.yml'

loaded_info = parse_config_for_analysis(config_path)


# ## Perplexity and Accuracy

# In[3]:


lm_RE = loaded_info["lightning_language_model_RE"].to("cpu")
lm = loaded_info["lightning_language_model_no_RE"].to("cpu")
tokenizer = loaded_info["tokenizer"]
dataloader_test = loaded_info["dataloader_test"]


# In[4]:


### First thing we compare the perplexity and accuracy on the testset.
from utils.analysis import get_results, get_results_RE


lm_RE_result = get_results_RE(lm_RE, dataloader_test, 5)
lm_result = get_results(lm, dataloader_test)
print(lm_RE_result)
print(lm_result)


# ## Change in perplexity

# In[6]:


### Next we test what happens if we check te change in perplexity of the RE. 
from utils.analysis import calc_change_in_perplexity_experiment
change_in_perplexity = calc_change_in_perplexity_experiment(lm_RE, dataloader_test, n_experiments=10, n_extra_mask=2)
change_in_perplexity


# ## Distribution of mask

# In[7]:


from utils.analysis import rational_analysis

rational_distributions = rational_analysis(lm_RE, dataloader_test)
print(rational_distributions)


# In[27]:


import matplotlib.pyplot as plt
relative_counts = rational_distributions["rel_pos_count"]
total = sum(relative_counts.values())
X = [int(k) for k in relative_counts.keys()] 
Y = [r/total for r in relative_counts.values()]

pairs = sorted([(x,y) for x,y in zip(X, Y)], key=lambda p: p[0])
plt.xlabel
X_sorted = [p[0] for p in pairs]
Y_sorted = [p[1] for p in pairs]
plt.xlabel("Relative Distance")
plt.ylabel("Percentage")
plt.plot(X_sorted, Y_sorted, "")


# In[29]:


plt.xlabel("Relative Distance")
plt.ylabel("Percentage")
plt.bar(X, Y)


# In[9]:


abs_pos_count = rational_distributions["abs_pos_count"]
plt.plt(abs_pos_count.keys(),abs_pos_count.values())


# ## Analysing some examples

# In[30]:


examples = ["How are you doing?", "What did you do today?", "How's work?", "Would you like some coffee?"]


# In[31]:


## First with greedy rationals
completed_dialogues_chance = lm_RE.complete_dialogues(examples, total_length=40, greedy_rationals=False)
completed_dialogues_greedy = lm_RE.complete_dialogues(examples, total_length=40, greedy_rationals=True)


# In[32]:


from utils.analysis import pretty_print_completed_dialogues


# In[33]:


pretty_print_completed_dialogues(completed_dialogues_greedy)


# In[34]:


pretty_print_completed_dialogues(completed_dialogues_chance)


# In[ ]:




