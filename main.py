# %%
"""
## **World Population Analysis**
"""

# %%
"""
## **Setting Up** <a class="anchor" id="su"></a>
"""

# %%
# setting up my environment

import numpy as np 
import pandas as pd 
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from itables import init_notebook_mode
from itables import show
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import plotly.offline as py
py.init_notebook_mode(connected=True)
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()

import warnings
warnings.filterwarnings('ignore')

# %%
"""
### **Importing Dataset** <a class="anchor" id="id"></a>
"""

# %%
df = pd.read_csv('./data/world_population.csv')
df.head()
df.tail(3)

# %%
"""
### **Data Processing** <a class="anchor" id="dp"></a>
"""

# %%
# renaming 'Country/Territory' to 'Country'
df.rename(columns={'Country/Territory':'Country'}, inplace = True)
df = df.drop_duplicates()
df.head()

# %%
# renaming year columns from "Year Population" to just "Year" 

for col in df.columns:
    if 'Population' and '0' in col:
        df = df.rename(columns={col: col.split(' ')[0]})
        
df.head(3)
df.tail(3)
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
df.info()

# %%
df.nunique()

# %%
# converting the column names into strings

df.columns = list(map(str, df.columns))

# %%
years = list(map(str, (1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022)))
years

# %%
df.describe().T.sort_values(ascending=0, by="mean")

# %%
continent_df = df.groupby(by='Continent').sum()
continent_df.head(3)

# %%
country_df = df.groupby(by='Country').sum()
country_df.head(3)

# %%
"""
### **Exploratory Data Analysis and Visualization** <a class="anchor" id="eda"></a>
"""

# %%
"""
### World Population EDA <a class="anchor" id="wpeda"></a>
"""

# %%
df['2022'].sum()

# %%
# plotting world population trend since 1970
plt.subplots(figsize=(10,5))
trend = df.iloc[:,5:13].sum()[::-1]
sns.lineplot(x=trend.index, y=trend.values, marker="o")
plt.xticks(rotation=45)
plt.ylabel("Population")
plt.title("World Population Trend (1970-2022)")
plt.show()

# animated world population trend since 1970
from matplotlib.animation import FuncAnimation
plt.rcParams['animation.html'] = 'jshtml'
plt.rcParams['figure.dpi'] = 100
plt.ioff()
fig, ax = plt.subplots(figsize=(10,5))
x = df.columns[5:13][::-1]
y = df.iloc[:,5:13].sum()[::-1]

def animate(i):
    plt.cla()
    ax.plot(x[:i], y[:i], marker="o")
    plt.xticks(rotation=45)
    plt.ylabel("Population (in billions)")
    plt.title('World population since 1970-2020')

ani = FuncAnimation(fig, animate, frames=np.arange(0, len(x)), interval=1000)
ani
# saving the animation as gif
ani.save('world_population.gif', writer='imagemagick')

# %%
#  plotting current world population on map

import pycountry
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3

    
# get average of a list
def Average(list):
    return sum(list) / len(list)

df_rc2022 = df.loc[:,["CCA3", "Country","2022"]]
df_rc2022["CCA3"] = [countries.get(x, 'Unknown code') for x in df_rc2022["Country"]]

fig = px.choropleth(df_rc2022, locations="CCA3",
                    hover_name="Country",
                    hover_data=df_rc2022.columns,
                    color="2022",
                    color_continuous_scale="Viridis",
                    range_color=(min(df_rc2022["2022"]), max(df_rc2022["2022"])), 
                    projection="natural earth"
                   
                   )

fig.update_layout(margin={"r":5,"t":0,"l":5,"b":0})
fig.show()

# %%
# let's see pie chart distribution for continent_df

continent_df['2022'].plot(kind = 'pie', figsize=(10,5), shadow=True, autopct='%1.1f%%') # autopct create %
plt.title(' Population Distribution by Continent')
plt.axis('equal')
plt.show()

# %%
# let's see world population by continent

fig = px.bar(data_frame= df.groupby('Continent' , as_index= False).sum().sort_values('2022', ascending=False), x= 'Continent' , y= '2022')
fig.update_layout(title= 'Current (2022) World Population per Continent', title_x= 0.5)
fig.show()

# %%
# let's see number of countries per continent

df_country=df['Continent'].value_counts()
fig=px.bar(x=df_country.index,
          y=df_country.values,
          color=df_country.index,
          color_discrete_sequence=px.colors.sequential.YlOrRd,
          text=df_country.values,
          title= 'Number of Countries By Continent')

fig.update_layout(xaxis_title="Countries",
                 yaxis_title="Count")

fig.show()

# %%
"""
### Population Growth Rate <a class="anchor" id="pgr"></a>
"""

# %%
# average population growth rate

df['Growth Rate'].mean()

# %%
"""
Since 1972 (50 years ago), the world population growth rate declined from around 2% per year to under 1.0% per year.
"""

# %%
# plotting population Growth Rate on map

fig = px.choropleth(df,
                     locations='Country',
                     locationmode='country names',
                     color='Growth Rate',
                     color_continuous_scale='Viridis',
                     template='plotly',
                     title = 'Growth Rate')

fig.update_layout(font = dict(size = 17, family="Gothic"))

# %%
# creating dataframe for top 10 countries with highest growth rate

gwr_top10 = df.sort_values(by='Country').sort_values(by='Growth Rate', ascending=False).head(10)

gwr_top10.head(3)

# %%
# plotting top 10 highest growth rate countries in the last 30 years

fig, ax = plt.subplots(figsize=(16,8))
plt.plot(gwr_top10['Country'], gwr_top10['2020'], label='2020', marker='o')
plt.plot(gwr_top10['Country'], gwr_top10['1990'], label='1990', marker='d')

plt.xlabel('Country')
plt.ylabel('Growth Rate')
plt.grid(linewidth=0.3)
plt.title('Top 10 Countries with Highest Growth Rate in the last 30 years')
plt.legend()
plt.show()

# %%
"""
### Population Decade-By-Decade Percent Change <a class="anchor" id="pdbdpc"></a>
"""

# %%
# creating dataframe for population difference decade-by-decade per continent

pop_diff = df.groupby('Continent')[['1970','1980', '1990', '2000', '2010', '2020']].sum().sort_values(by='Continent').reset_index()

pop_diff.head(3)

# %%
# finding the population decade-by-decade percent change

pop_diff['70s'] = pop_diff['1970']/pop_diff['1980']*100
pop_diff['80s'] = pop_diff['1980']/pop_diff['1990']*100
pop_diff['90s'] = pop_diff['1990']/pop_diff['2000']*100
pop_diff['00s'] = pop_diff['2000']/pop_diff['2010']*100
pop_diff['10s'] = pop_diff['2010']/pop_diff['2020']*100

pop_diff.head(3)

# %%
# creating dataframe for decade-by-decade

decade_diff = pop_diff.groupby('Continent')[['70s','80s', '90s', '00s', '10s']].sum().sort_values(by='Continent').reset_index()

decade_diff

# %%
# let's see decade_diff statistical summary quickly

decade_diff.describe()

# %%
# plotting wolrd population difference decade-by-decade percent change 70s - 2010s

fig, ax = plt.subplots(figsize=(16,8))
plt.plot(decade_diff['Continent'], decade_diff['70s'], label='70s', marker='o', color='green')
plt.plot(decade_diff['Continent'], decade_diff['80s'], label='80s', marker='d', color='red')
plt.plot(decade_diff['Continent'], decade_diff['90s'], label='90s', marker='o', color='blue')
plt.plot(decade_diff['Continent'], decade_diff['00s'], label='00s', marker='d', color='orange')
plt.plot(decade_diff['Continent'], decade_diff['10s'], label='10s', marker='d', color='skyblue')
plt.grid(linewidth=0.4)
plt.title("World Population Difference Decade-By-Decade Percent Change 1970s to 2010s")
plt.xlabel('Continents')
plt.ylabel('Population')
plt.legend()
plt.show()

# %%
"""
### Area <a class="anchor" id="area"></a>
"""

# %%
# plotting total area distribution by continents

import plotly.graph_objects as go

df_cont= df['Continent'].unique()

tot_area_cont = []

for each in df_cont:
    df_area = df[df.Continent == each]
    area = sum(df_area["Area (km²)"])
    tot_area_cont.append(area)
    
tot_area_cont = pd.DataFrame(tot_area_cont)
df_area = pd.DataFrame(df_cont, columns = ["continent"])
df_area["total"] = tot_area_cont

fig = go.Figure(data=[go.Pie(labels=df_area.continent, values=df_area.total, textinfo='label+percent',
                             insidetextorientation='radial'
                            )])
fig.show()

# %%
# plotting Area distribution on map by country

fig = px.choropleth(df,
                     locations='Country',
                     locationmode='country names',
                     color='Area (km²)',
                     color_continuous_scale='Viridis',
                     template='plotly',
                     title = 'Area (km²)')

fig.update_layout(font = dict(size = 17, family="Gothic"))

# %%
"""
### Top 10 Countries With Most Population <a class="anchor" id="t10mp"></a>
"""

# %%
"""
Firstly, let's copy our dataframe 'df' to 'df_copy' so we can place 'Country' as the index to avoid affecting other analysis negatively.
"""

# %%
# copying dataframe 'df' to 'df_copy'

df_copy = df.copy()
df_copy.head(3)

# %%
df_copy.set_index('Country', inplace=True)

# %%
df_copy.sort_values(by='2022', ascending=True, inplace=True)

df_top10 = df_copy['2022'].tail(10)
df_top10

# %%
# plotting top 10 MOST populated countries

df_top10.plot(kind='barh', figsize=(10, 10), color='darkblue')
plt.xlabel('Population')
plt.title('Top 10 Countries With MOST Population 2022')

plt.show()

# %%
# plotting top 10 population trend

inplace = True 
df_copy.sort_values(by='2022', ascending=False, axis=0, inplace=True)

df_top_10 = df_copy.head(10)

df_top_10 = df_top_10[years].transpose() 

df_top_10.index = df_top_10.index.map(int)
df_top_10.plot(kind='line', figsize=(14, 8)) 

plt.title('Trend of Top 10 MOST Populated Countries')
plt.ylabel('Population')
plt.xlabel('Years')
plt.show()

# %%
"""
### Top 10 Countries With Least Population <a class="anchor" id="t10lp"></a>
"""

# %%
df_copy.sort_values(by='2022', ascending=True, inplace=True)

df_btm10 = df_copy['2022'].head(10)
df_btm10

# %%
df_btm10.plot(kind='barh', figsize=(10, 10), color='darkred')
plt.xlabel('Population')
plt.title('Top 10 Countries With LEAST Population 2022')

plt.show()

# %%
#  top 10 countries with least population trend.

inplace = True 
df_copy.sort_values(by='2022', ascending=False, axis=0, inplace=True)

df_bttm10 = df_copy.tail(10)

df_bttm10 = df_bttm10[years].transpose() 

df_bttm10.index = df_bttm10.index.map(int)
df_bttm10.plot(kind='line', figsize=(14, 8)) 

plt.title('Trend of Top 10 Countries with LEAST population')
plt.ylabel('Populaton')
plt.xlabel('Years')
plt.show()

# %%
"""
### Top 5 Most Populated Countries By Continents <a class="anchor" id="t5mp"></a>
"""

# %%
"""
Here, we don't need to copy our dataframe or use the copied dataframe. We will use our original dataframe 'df' in this section
"""

# %%
# creating dataframes for countries per continent

# Asia
asian_countries = df.loc[df["Continent"]=="Asia"].sort_values(by=["2022"], ascending=False, ignore_index=True)

# Africa
african_countries = df.loc[df["Continent"]=="Africa"].sort_values(by=["2022"], ascending=False, ignore_index=True)

# Europe
european_countries = df.loc[df["Continent"]=="Europe"].sort_values(by=["2022"], ascending=False, ignore_index=True)

# North America
na_countries = df.loc[df["Continent"]=="North America"].sort_values(by=["2022"], ascending=False, ignore_index=True)

# Oceania
oc_countries = df.loc[df["Continent"]=="Oceania"].sort_values(by=["2022"], ascending=False, ignore_index=True)

# South America
sa_countries = df.loc[df["Continent"]=="South America"].sort_values(by=["2022"], ascending=False, ignore_index=True)


# %%
# plotting top 5 MOST populated countries by continent

# Asian countries
asian_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="Asia Top 5 MOST Populated Countries", figsize=(8,6), color = 'skyblue', fontsize=12)

# African countries
african_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="Africa Top 5 MOST Populated Countries", figsize=(8,6), color = 'darkgreen', fontsize=12)

# European countries
european_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="Europe Top 5 MOST Populated Countries", figsize=(8,6), color = 'orange', fontsize=12)

# North American countries
na_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="North America Top 5 MOSTG Populated Countries", figsize=(8,6), color = 'darkred', fontsize=12)

# Oceanian countries
oc_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="Oceania Top 5 MOST Populated Countries", figsize=(8,6), color = 'purple', fontsize=12)

# South American countries
sa_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).head(5).plot.bar(x="Country", ylabel="Population", title="South America Top 5 MOST Populated Countries", figsize=(8,6), color = 'black', fontsize=12)


# %%
"""
### Top 5 Least Populated Countries By Continents <a class="anchor" id="t5lp"></a>
"""

# %%
# plotting top 5 LEAST populated countries by continent

# Asian countries
asian_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="Asia Top 5 LEAST Populated Countries", figsize=(8,6), color = 'skyblue', fontsize=12)

# African countries
african_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="Africa Top 5 LEAST Populated Countries", figsize=(8,6), color = 'darkgreen', fontsize=12)

# European countries
european_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="Europe Top 5 LEAST Populated Countries", figsize=(8,6), color = 'orange', fontsize=12)

# North American countries
na_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="North America Top 5 LEAST Populated Countries", figsize=(8,6), color = 'darkred', fontsize=12)

# Oceanian countries
oc_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="Oceania Top 5 LEAST Populated Countries", figsize=(8,6), color = 'purple', fontsize=12)

# South American countries
sa_countries[["Country", "2022"]].sort_values(by="2022", ascending=False).tail(5).plot.bar(x="Country", ylabel="Population", title="South America Top 5 LEAST Populated Countries", figsize=(8,6), color = 'black', fontsize=12)


# %%
"""
### Population Projection <a class="anchor" id="pp"></a>
"""

# %%
"""
### World 2030 Population Projection <a class="anchor" id="w230pp"></a>
"""

# %%
# current world population

df['2022'].sum()

# %%
"""
World current population: 7.9 billion
"""

# %%
# relationship betewen years and total population, we will convert years to int type.

df_tot = pd.DataFrame(df[years].sum(axis=0)) # use the sum() method to get the total population per year

df_tot.index = map(int, df_tot.index) # change the years to type int (useful for regression later on)

df_tot.reset_index(inplace = True) # reset the index to put in back in as a column in the df_tot dataframe

df_tot.columns = ['year', 'total'] # rename columns

df_tot.head()

# %%
# plotting a scatter plot for year vs total population

df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='black')

plt.title('Current Wolrd Population')
plt.xlabel('Year')
plt.ylabel('Population')

plt.show()

# %%
# fitting our data

x = df_tot['year']      
y = df_tot['total']    
fit = np.polyfit(x, y, deg=1)

fit

# %%
# plotting the regression line on the scatter plot

df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='black')

plt.title('World Population 1970 - 2022')
plt.xlabel('Year')
plt.ylabel('Population')

# plot line of best fit
plt.plot(x, fit[0] * x + fit[1], color='red') # recall that x is the Years
plt.annotate('y={0:.0f} x + {1:.0f}'.format(fit[0], fit[1]), xy=(2000, 150000))

plt.show()

# print out the line of best fit
'World Population = {0:.0f} * Year + {1:.0f}'.format(fit[0], fit[1]) 


# predicting the world population in 2030
from sklearn.linear_model import LinearRegression

# create linear regression object
lm = LinearRegression()
lm

# fit the data
lm.fit(df_tot[['year']], df_tot[['total']])

# predicting the world population in 2030
x = 2030
y = lm.predict([[x]])
print('The world population in 2030 will be {0:.0f}'.format(y[0][0]))

# Plotting the data and the regression line
new_df = pd.DataFrame({'year': [x], 'total': [y]})
new_df

df_tot = pd.concat([df_tot, new_df], ignore_index=True)
df_tot


# plotting the data and the regression line
df_tot.plot(kind='scatter', x='year', y='total', figsize=(10, 6), color='black')

plt.title('World Population 1970 - 2030 prediction')
plt.xlabel('Year')
plt.ylabel('Population')

plt.plot(df_tot['year'], lm.predict(df_tot[['year']]), color='red')

plt.show()
