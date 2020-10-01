import bamboolib as bam
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import seaborn
import matplotlib.dates as mdates
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from scipy import stats
import researchpy as rp
from scipy.stats import chisquare

# file to get county in TX classification
df2 = pd.read_excel('PHR_MSA_County_masterlist.xlsx').truncate(before=0, after=253)
df2['NCHS Urban Rural Classification (2013)'] = df2['NCHS Urban Rural Classification (2013)'].replace('Micropolitan',1).replace('Large Central Metro',2).replace('Large Fringe Metro',3).replace('Medium Metro', 4).replace('Small Metro', 5).replace('Non-core', 6)
df2['Classication (2013)'] = df2['NCHS Urban Rural Classification (2013)'].astype('Int64')

# file to relate FIPS to df2 to plot county distribution map
df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv')
df_sample_r = df_sample[df_sample['STNAME'] == 'Texas']
df_sample_r = df_sample_r.reset_index(drop=True)


# get FIPS data from df_sample_r
i = 0
for county in df2['FIPS #']:
    df2['FIPS #'] = df2['FIPS #'].replace(df2['FIPS #'][i], df_sample_r['FIPS'][i])
    i += 1
df2['FIPS #'] = df2['FIPS #'].astype('int64')


# plot county classification
fips = df2['FIPS #']
colorscale = ["#08306b", "#2171b5", "#57a0ce","#9ecae1", "#d2e3f3", "#f7fbff"]

fig = ff.create_choropleth(
    fips=fips, values=df2['Classication (2013)'], scope=['TX'],
    binning_endpoints=[1, 2, 3, 4, 5],
    colorscale=colorscale,
    county_outline={'color': 'rgb(28, 28, 28)', 'width': 0.5},
    round_legend_values=True,
    legend_title = 'County Classication',
)
fig.layout.template = None
plt.savefig('county classification', dpi=60)
# 'Micropolitan',<1
# 'Large Central Metro',1-2
# 'Large Fringe Metro',2-3
# 'Medium Metro',3-4
# 'Small Metro',4-5
# 'Non-core', >5


# get rural area from df2
rural = []
n = 0
for county in df2['County Name']:
    if df2['Classication (2013)'][n] == 6:
        rural.append(df2['County Name'][n])
    n += 1
# print(rural)


# get urban area from df2
def urbanarea(m):
    # urban=defaultdict(list)
    urban = []
    n = 0
    for county in df2['County Name']:
        if df2['Classication (2013)'][n] == m:
            urban.append(df2['County Name'][n])
            # urban['urbanarea{}'.format(m)].append(df2['County Name'][n])
        n += 1
    return urban
    # print(urbanarea(2))


# get rural neiboring the urban
# df3 county adjency file
df3 = pd.read_excel('county_adjacency.xlsx').truncate(before=17472, after=19249)
df3 = df3.reset_index(drop=True)
df3['county'] = df3['county'].replace(np.nan, 'n')


def countyadj(c):
    i = 0
    # adj = defaultdict(list)
    adj = []
    for county in df3['county']:
        if c in df3['county'][i]:
            # print(df3['county'][i])
            # adj[c].append(df3['countyadj'][i])
            adj.append(df3['countyadj'][i])
            j = i + 1
            for county in df3['county'][j:]:
                if df3['county'][j] != 'n':
                    break
                if df3['county'][j] == 'n':
                    # adj[c].append(df3['countyadj'][j])
                    adj.append(df3['countyadj'][j])
                j += 1
        i += 1
    return (adj)
    # print(adj)


def ruraladj_urban(j):
    countyadj_urban = []
    for a in urbanarea(j):
        countyadj_urban.extend(countyadj(a))

    ruraladj_urban = []
    # only counties which belongs to rural
    i = 0
    for county in countyadj_urban:
        if countyadj_urban[i].split()[0] in rural:
            ruraladj_urban.append(countyadj_urban[i].split()[0])
        i += 1
    return (ruraladj_urban)
# print(ruraladj_urban(2))


# here, need get 5 class, store in one list, each sublist for one class
ruralurban = []
for i in range(1,6):
    ruralurban.append(list(set(ruraladj_urban(i))))
print('rural neiboring 5 urban counties:', ruralurban)


rural_nonadjurban = set(rural) - set(ruraladj_urban(1)) - set(ruraladj_urban(2)) - set(ruraladj_urban(3)) - set(
    ruraladj_urban(4)) - set(ruraladj_urban(5))
print('rural not neiboring urban:', rural_nonadjurban)


# clean case data for data analysis
url = "https://dshs.texas.gov/coronavirus/TexasCOVID19DailyCountyCaseCountData.xlsx"  # sheets 'COVID-19 Cases'
df = pd.read_excel(url, sheet_name='Cases by County', skiprows=2)
df = df.truncate(before=0, after=253)
df = df.set_index('County Name')

pattern = re.compile('[0-9]+-[0-9]+', re.IGNORECASE)
dates = [datetime.strptime(pattern.findall(sub)[0], '%m-%d') for sub in df.keys()]
df.columns = [pattern.findall(sub)[0] for sub in df.keys()]
mydf = df.T.copy()
mydf = mydf.diff()
mydf = mydf.replace(np.nan, 0)


# get population to calculate rate in per
def get_pop(o):
    pop = 0
    i = 0
    for county in df_sample_r['CTYNAME']:
        if o in df_sample_r['CTYNAME'][i]:
            pop = df_sample_r['TOT_POP'][i]
            break
        i += 1
    return pop
    #print(pop)


# #t test between rural near urban1 and non-neiboring urban rural area
# ttest = []
# for a in rural_nonadjurban:
#     popa = get_pop(a)
#     for b in ruraladj_urban1:
#         popb = get_pop(b)
#         r = stats.ttest_ind(mydf[a] / popa, mydf[b] / popb)
#         ttest.append(r)
#         print(ttest)
# print(ttest)
#
# #z-test between rural near urban1 and non-neiboring urban rural area
# from statsmodels.stats import weightstats as stets
# ztest = []
# for a in rural_nonadjurban:
#     popa = get_pop(a)
#     for b in ruraladj_urban1:
#         popb = get_pop(b)
#         r = stets.ztest(mydf[a] / popa, mydf[b] / popb, value=0, alternative='two-sided')
#         ztest.append(r)
#         print(ztest)

# do test between every county in two class seems hard, out of memery


# rate for non-neiboring urban rural area
mydf['rate'] = 0
for county in rural_nonadjurban:
    mydf['rate'] = mydf['rate'] + mydf[county]/get_pop(county)
rate = mydf['rate']/len(rural_nonadjurban)
# print(rate)


# rate for other 5 classes near 5 urban areas
def ra(i):
    mydf['i'] = 0
    for county in ruralurban[i - 1]:
        mydf['i'] = mydf['i'] + mydf[county] / get_pop(county)
    ra = mydf['i'] / len(ruralurban[i - 1])

    # store rate for 5 class in mydf
    if i == 1:
        mydf['ra1'] = mydf['i'] / len(ruralurban[i - 1])
    elif i == 2:
        mydf['ra2'] = mydf['i'] / len(ruralurban[i - 1])
    elif i == 3:
        mydf['ra3'] = mydf['i'] / len(ruralurban[i - 1])
    elif i == 4:
        mydf['ra4'] = mydf['i'] / len(ruralurban[i - 1])
    elif i == 5:
        mydf['ra5'] = mydf['i'] / len(ruralurban[i - 1])

    return ra
# print(ra(1))


# chi-square test
for j in range(1,6):
    crosstab, test_results = rp.crosstab(rate, ra(j),test= "chi-square")
    print('chi-square test:', test_results)


# plot scatter
from pandas import plotting
import matplotlib.pyplot as plt
plotting.scatter_matrix(mydf[['rate', 'ra1', 'ra2','ra3','ra4','ra5']])
plt.rcParams['figure.figsize']=(20,16)
plt.show()


# specify an OLS model and fit it
from statsmodels.formula.api import ols
for j in range(1, 6):
    data = pd.DataFrame({'x': rate, 'y': ra(j)})
    model = ols("y ~ x", data).fit()
    print(model.summary())


# plot regression
g=seaborn.pairplot(mydf, vars=['rate', 'ra1', 'ra2','ra3','ra4','ra5'],kind='reg')
g.set(xscale='log')
plt.show()


# note:
# t-test, z-test and ANOVA are based on mean of data, doesn't apply to our case for
# a time series rate analysis, rate in March have a big difference with rate in Aug