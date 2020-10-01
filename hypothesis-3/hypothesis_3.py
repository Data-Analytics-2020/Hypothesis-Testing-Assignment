#!/usr/bin/env python
# coding: utf-8

# # Predefined functions

# In[1]:


import pandas as pd

# Getting data
def getData(url, sheet_name=0, skiprows=0, truncate=-1, index_column='County Name'):
    df = pd.read_excel(url, sheet_name=sheet_name, skiprows=skiprows)
    if truncate > -1:
        df = df.truncate(before=0, after=truncate)
    df = df.set_index(index_column)
    return df


# # Data collection

# ## Cumulative case count

# In[2]:


url = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx"
cases_df = getData(url, skiprows=2, truncate=254)


# In[3]:


import re
from datetime import datetime
pattern = re.compile('[0-9]+-[0-9]+', re.IGNORECASE)
cases_df.columns = [pattern.findall(sub)[0] for sub in cases_df.keys()]
cases_df.columns = [datetime.strptime(date, '%m-%d') for date in cases_df.keys()]


# # Plotting

# In[4]:


import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.dates as mdates

figsize=(12,8)
title_size = 24
label_size = 18
ticker_count= 8
line_width = 2.5


# ## 7-day average daily case count

# In[5]:


import seaborn as sns
from datetime import datetime, timedelta

# create figure
fig, ax = plt.subplots(figsize=figsize)
fig.suptitle('New daily COVID-19 cases in Texas', fontsize=title_size)

data = cases_df.loc['Total']
ylim = 20000
# new daily case
daily = data.diff()
sns.lineplot(label="Daily", data=daily, ci=None, color='pink', ax=ax)
# 7-day rolling average
avg7day = daily.rolling(window=7).mean()
g = sns.lineplot(label="7-day average", data=avg7day, color='crimson')

# major events
events = {'Disaster':'Apr 12', 'Phase 1':'May 1', 'Phase 2':'May 18', 'Phase 3':'Jun 3', 'Paused':'Jun 25', 'Mask requried':'Jul 2'}
for event, date in events.items():
    date_obj = datetime.strptime(date, '%b %d')
    plt.axvline(date_obj, color='gray', linewidth=3)
    plt.text(date_obj + timedelta(days=2), ylim*0.7, event + ': ' + date, rotation=90, fontsize=13)

# fill between 2 lines
ax.fill_between(data.keys(), daily, avg7day, facecolor='lavenderblush')
    
# legend
g.legend(loc='upper right')
    
# ylim
ax.set_xlim(cases_df.keys()[0], cases_df.keys()[-15])
ax.set_ylim(0, ylim)

# horizontal grid lines
ax.yaxis.grid(True)

# remove outer frame
plt.box(on=None)

# number of tickers
ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
# ticker formatter
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

# configure legend on lines
for line, name in zip(ax.lines, ['Daily', '7-day']):
    y = line.get_ydata()[-1]
    ax.annotate(name, xy=(1,y), xytext=(6,0), color=line.get_color(), 
                xycoords = ax.get_yaxis_transform(), textcoords="offset points",
                size=18, va="center")

fig.savefig('../graphs/hypothesis_3.png')

