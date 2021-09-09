#!/usr/bin/env python
# coding: utf-8

# # Problem : 
# ## Uber Facility Enbles Customers to Get rides at reasonable rates BUT a Cab drivers faces problem of taking customers which are at farther distance from them. This is because, the cab drivers have to use their own fuel to get to their respective passengers and fuel rates these days are getting increased day by day.
# <br>
# <br>
# 
# # Solution :
# ## Analysing Dataset of Uber Cab facility of a location and concluding a convenient place  where we can create hubs of cabs or garage. This will enble the drivers to get fare from nearby locations from the hubs created by MACHINE LEARNING ALGORITHM

# ## Importing necessary libraries and Using Dataset

# In[99]:


import pandas as pd                # for dataset evaluation
import numpy as np                 # for dataset evaluation
from datetime import datetime      # for date and time analysis 
import matplotlib.pyplot as plt    # for dataset visualisation
get_ipython().run_line_magic('matplotlib', 'inline')

# %matplotlib inline sets the backend of matplotlib to the 'inline' backend: 
# With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, 
# directly below the code cell that produced it


# In[142]:


# Datasets for the 6 month period is present in 6 different files for the New York City
# Read the data set separately 

apr = pd.read_csv("uber-raw-data-apr14.csv")
may = pd.read_csv("uber-raw-data-may14.csv")
jun = pd.read_csv("uber-raw-data-jun14.csv")
jul = pd.read_csv("uber-raw-data-jul14.csv")
aug = pd.read_csv("uber-raw-data-aug14.csv")
sep = pd.read_csv("uber-raw-data-sep14.csv")


# ## Exploratory Data Analysis

# In[181]:


# combine all data to a single dataset

data1 = [apr,may,jun,jul,aug,sep]
data1 = pd.concat(data1)
data1


# In[160]:


data1.head() #seing dataframe's top 5 rows


# In[161]:


data1.columns #knowing columns and their types


# In[162]:


data1.shape  # 4534327 X 4 = 1,81,37,308 entries in dataset


# In[163]:


data1.dtypes  #datatype of different entries


# In[182]:


data1.info() 


# In[165]:


data1.describe()


# In[166]:


# Disaggregate 'Date/Time' column and converting it to PYTHON date and time object

data1['Date/Time'] = pd.to_datetime(data1['Date/Time'])
data1['Month'] = data1['Date/Time'].dt.month
data1['Day'] = data1['Date/Time'].dt.day
data1['Time'] = data1['Date/Time'].dt.time

data1.head() # visualising after object encoding 


# It would also be useful to know the **day of the week** in which each pickup occurred. 
# Additionally, the __Time__ column is quite granular, so let's also create an __Hour__ column.

# In[168]:


# Day of the week
data1['Weekday'] = data1['Date/Time'].dt.dayofweek

# Hour
data1['Hour'] = data1['Date/Time'].dt.hour

data1.head()


# Hear hour is __0__ because we _don't have the travel duration or destination reaching time in dataset_. <br>
# _We __don't need Hour__ column for any analysis_

# In[169]:


# Map month decoding form numeral value to String Month/Day Name using dictionary
month_map = {
    4: 'April',
    5: 'May',
    6: 'June', 
    7: 'July',
    8: 'August',
    9: 'September'
}
 # Can include the following months if we have the dataset
 #   1: 'January',
 #   2: 'February',
 #   3: 'March',
    
 #   10: 'October',
 #   11: 'November',
 #   12: 'December',

data1['Month'] = data1['Month'].replace(month_map)

# Map weekday
weekday_map = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday', 
    4: 'Friday', 
    5: 'Saturday',
    6: 'Sunday'
}
data1['Weekday'] = data1['Weekday'].replace(weekday_map)

final_columns = ['Base', 'Lat', 'Lon', 'Month', 'Day', 'Weekday']

data1 = data1[final_columns]

data1.head()


# In[170]:


# Checking for duplicate data

duplicate_pickups = data1[data1.duplicated(keep=False)]

duplicate_pickups


# In[174]:


# Daily Average Pickups

num_pickups = data1.shape[0]
num_days = len(data1[['Month', 'Day']].drop_duplicates())
daily_avg = np.round(num_pickups/num_days, 0)

stats_raw = 'Number of Pickups:\t {}\nNumber of Days:\t\t {}\nAverage Daily Pickups:\t {}'
print(stats_raw.format(num_pickups, num_days, daily_avg))


# ## Data Visualisation

# ### Monthly UBER pickups

# In[176]:


monthly_pickups = data1['Month'].value_counts(ascending=True)[month_map.values()]

monthly_pickups.plot(kind='bar', rot=0)
plt.title('Uber Pickups Per Month')
plt.xlabel('Month')
plt.ylabel('# Pickups (Millions)')


# ### Weekdays UBER pickups

# In[177]:


weekday_pickups = data1['Weekday'].value_counts()[weekday_map.values()]

weekday_pickups.plot(kind='bar', rot=0)
plt.title('Uber Pickups Per Weekday')
plt.xlabel('Weekday')
plt.ylabel('# Pickups')


# In[175]:


# Comparsion of Weekdays and monthly data using bar graph

monthly_weekdays = data1.groupby('Month')['Weekday'].value_counts().unstack()
monthly_weekdays_norm = monthly_weekdays.apply(lambda x: x/x.sum(), axis=1)

monthly_weekdays_norm.loc[month_map.values(),weekday_map.values()].plot(kind='bar', rot=0)
plt.ylabel('Proportion of Pickups')
plt.title("Uber Pickups by Month and Weekday")


# In[178]:


# Understanding importance of Base in dataset.

base_props = data1['Base'].value_counts(normalize=True)
display(base_props)

base_props.plot(kind='bar', rot=0)
plt.xlabel('Base')
plt.ylabel('Proportion of Pickups')
plt.title('Pickups by Base');


# There are mainly 6 categories of BASE which determine the TLC (Taxi and Limousine Commission) base __company code__ affiliated with the Uber pickup. <br>
# _It is too __not required__ in the dataset_

# In[179]:


# Trend in dataset with months for different kind of Uber Cabs(BASES)

monthly_bases = data1.groupby('Month')['Base'].value_counts().unstack()

monthly_bases.loc[month_map.values()].plot(kind='line', marker='o', rot=0)
plt.ylabel('# Pickups')
plt.title('Uber Pickups by Month and Base');


# ### It was concluded that dataset for the September month was having the highest trend and weightage amongst the datasets. <br>
# ## Dataset for September was chosen for designing Machine Learning Algorithm.
# uber-raw-data-sep14.csv

# In[183]:


#Uber_dataset selected

data=pd.read_csv("uber-raw-data-sep14.csv")


# In[184]:


data


# ### Date/Time: The date and time of a CAB pickup.
# ### Lat(Latitude): The latitude of the CAB pickup (in degrees)
# ### Lon(Longitude): The longitude of the Uber pickup. (in degrees)
# ### Base: The TLC (Taxi and Limousine Commission) base _*company code*_ affiliated with the Uber pickup.

# In[185]:


data.info()


# In[187]:


data.describe()


# In[188]:


data.columns


# # Data Visualisation For Selected Data

# In[190]:


# Kernel Density Estimation Graph for Latitudes

data["Lat"].plot.kde()


# In[10]:


# Kernel Density Estimation Graph for Longitudes

data["Lon"].plot.kde()


# In[127]:


# Plotting of possible graphs

import seaborn as sns
sns.pairplot(data)


# # K-Means is Selected for building the Model
# 
# #### k-means minimizes within-cluster variance, which equals squared Euclidean distances.
# In general, the arithmetic mean does this. It does not optimize distances, but squared deviations from the mean.
# 
# #### k-medians minimizes absolute deviations, which equals Manhattan distance.
# In general, the per-axis median should do this. It is a good estimator for the mean, if you want to minimize the sum of absolute deviations (that is sum_i abs(x_i-y_i)), instead of the squared ones.
# 
# ##### It's not a question about accuracy. It's a question of correctness.
# 
# So here's your decision tree:
# 
# If your distance is squared Euclidean distance, use k-means <BR>
# If your distance is Taxicab metric, use k-medians <BR>
# If you have any other distance, use k-medoids <BR>
#     
# Ref: https://stats.stackexchange.com/questions/109547/k-means-vs-k-median
# 
# #### Hence K-Means suits this dataset.

# In[191]:


# Segregating Latitude and longitude

groups=data[['Lat','Lon']]


# In[129]:


# Perfoming K-mean clustering for a sample value of 2 clusters to see the inertia of 2 clusters

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans.fit(groups)


# In[130]:


kmeans.inertia_  #inertia for k value = 2


# Inertia can be _recognized as a measure of how internally coherent clusters are_. It suffers from various __drawbacks__:
# 
# Inertia makes the assumption that clusters are convex and isotropic, which is not always the case. It responds poorly to elongated clusters, or manifolds with irregular shapes.
# 
# Inertia is not a normalized metric: we just know that lower values are better and zero is optimal. But in very high-dimensional spaces, Euclidean distances tend to become inflated (this is an instance of the so-called “curse of dimensionality”). Running a dimensionality reduction algorithm such as Principal component analysis (PCA) prior to k-means clustering can alleviate this problem and speed up the computations.
# 
# Ref: https://scikit-learn.org/stable/modules/clustering.html

# In[192]:


# To store the values of different Sum of Square distances for range of clusters from 1 to 20

SSDist = [] 
for cluster in range(1,20):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(groups)
    SSDist.append(kmeans.inertia_)

    
    
# Converting the results into a dataframe and plotting them to get an ELBOW Plot and hence determine the target value of k  

frame = pd.DataFrame({'Cluster':range(1,20), 'SSDist':SSDist})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSDist'], marker='o')
plt.xlabel('Value of K')
plt.ylabel('Inertia')

    


# In[193]:


# The Value of K from the elbow plot which should be good for the model was found to be 6

kmeans=KMeans(n_clusters=6, init='k-means++')
kmeans.fit(groups)


# In[194]:


# The values of Centroid of 6 clusters formed

centroids=kmeans.cluster_centers_
centroids


# In[195]:


# Converting centroid in dataframe. These are nothing but the Lattitudnal and Longitudnal Co-ordinates

clocation=pd.DataFrame(centroids)
clocation


# In[196]:


# Plotting the Centroids in a scatter graph

plt.scatter(clocation[0],clocation[1],marker="o", color='Red', s=200)
plt.show()

# No proper relevance or trend is shown in the scatter plot.
# There seems to be 2 outliers (1st and 3rd points) but these can't be ignored. They too have an important role.
# We need to visualise the co-ordinates in the google map


# # Import gmplot, folium
# 
# For Jupiter Notebook: <br>
# Use Anaconda Prompt to run the following commands for google map dependency. <br>
# _You need to have Git installed on your PC for cloning gmplot repository_ <br>
# 1. <br>
# conda install -c mlgill gmplot <br>
# 2. <br>
# anaconda search -t conda gmplot <br>
# 3. <br>
# pip install gmplot <br>
# <br>
# pip install folium <br>
# <br>
# 
# _4. {Not required}_ <br>
# 
# *For local installation we need to clone the Github repository, use the following command: <br> 
# git clone https://github.com/vgm64/gmplot <br>
# 
# Now, you have the dependency on your PC at your Jupiter Notebook's Directory. <br> 
# Locate setup.py in gmplot folder of your Jupiter Notebook's Directory. <br>
# Run, <br>
# 
# python setup.py install *
# _
# 
# _Restart_ your Jupiter Notebook's Shell
# 
# Now you can import folium
# 

# In[135]:


import folium


# In[197]:


# Converting centroid into lists

centroid = clocation.values.tolist()


# Plotting the centroids on google map using Folium library.
map = folium.Map(location=[40.797981, -73.8749402], zoom_start = 10
                )
for point in range(0, len(centroid)):
    folium.Marker(centroid[point], popup = centroid[point]).add_to(map)
map


# In[224]:


# Checking for a Value in the clusters formed by the Algorithm

new_location=[(40.60,-73.74)]

result=kmeans.predict(new_location)
location=result[0]
result_location=clocation.iloc[location]
mark=list(result_location)
mark


# In[225]:


m = folium.Map(location=mark, zoom_start=11)

folium.Marker(
    location=[40.60,-73.74],
    popup="You are Here",
    icon=folium.Icon(color="purple"),
).add_to(m)

folium.Marker(
    location=mark,
    popup="Your Ride is Here",
    icon=folium.Icon(color="blue"),
).add_to(m)
m


# #  <center> END of PROJECT-1 for Uber Dataset in New York </center>

# # Challenges and Improvements
# 
# 1. Uber Dataset for different months in India is not available.
# 2. Removing of incorrect clusters such as that within water bodies to the nearest landmass.
# 3. Checking for other values from a sample datset.
# 
