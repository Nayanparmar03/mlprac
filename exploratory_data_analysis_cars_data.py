-- using EDA Cars.xlsx


## Exploratory Data Analysis (EDA)

Objective to achieve, understand and implement following topics

– Handle Missing value

– Removing duplicates

– Outlier Treatment

– Normalizing and Scaling( Numerical Variables)

– Encoding Categorical variables( Dummy Variables)

– Bivariate Analysis
"""

# Commented out IPython magic to ensure Python compatibility.
# Importing all the necessary libraries
import numpy as np   # numerial python
import pandas as pd  # dataframe working
import matplotlib.pyplot as plt #  plotting
import seaborn as sn # advance plotting
# %matplotlib inline

df_car = pd.read_excel("/content/EDA Cars.xlsx")
df_car.head(30)

df_car.shape

df_car.info()

df_car.describe()

df_car.describe(include=['object', 'int','float'])

df_car.INCOME

df_car[df_car['INCOME'] == '@@']



df_car['INCOME'] = df_car['INCOME'].replace(to_replace = '@@', value= np.nan)
df_car['INCOME'] = df_car['INCOME'].astype(float)

df_car[df_car['INCOME'] == '@@']

#df_car['TRAVEL TIME'] = df_car['TRAVEL TIME'].replace(to_replace = '*****', value = np.nan)

df_car['TRAVEL TIME'] = df_car['TRAVEL TIME'].astype(float)

#df_car['MILES CLOCKED'] = df_car['MILES CLOCKED'].replace(to_replace = 'Na', value = np.nan)
#df_car['MILES CLOCKED'] = df_car['MILES CLOCKED'].replace(to_replace = '*****', value = np.nan)
#df_car['MILES CLOCKED'] = df_car['MILES CLOCKED'].astype(float)

#df_car.replace(to_replace='*****', value= np.nan, inplace = True)

df_car.info()

# Check for missing values in any column or Handling missing values in dataset

df_car.isnull().sum()

"""### We can see that we have various missing values in the respective columns. There are various ways of treating your missing values in the data set. And which technique to use when is actually dependent on the type of data you are dealing with.

* Drop the missing values: In this case, we drop the missing values from those variables. In case there are very few missing values you can drop those values.

* Impute with mean value: For the numerical column, you can replace the missing values with mean values. Before replacing with mean value, it is advisable to check that the variable shouldn’t have extreme values .i.e. outliers.

* Impute with median value: For the numerical column, you can also replace the missing values with median values. In case you have extreme values such as outliers it is advisable to use the median approach.

* Impute with mode value: For the categorical column, you can replace the missing values with mode values i.e the frequent ones.
"""

# In this exercise, we will replace the numerical columns with median values and for categorical columns, we will drop the missing values.


#Replacing the NULL Values in numerical columns using MEDIAN
median_income = df_car['INCOME'].median()
median_travel_time = df_car['TRAVEL TIME'].median()
median_miles_clocked = df_car['MILES CLOCKED'].median()
median_car_age = df_car['CAR AGE'].median()
median_postal_code = df_car['POSTAL CODE'].median()


df_car['INCOME'].replace(np.nan, median_income, inplace = True)
df_car['TRAVEL TIME'].replace(np.nan, median_travel_time, inplace = True)
df_car['MILES CLOCKED'].replace(np.nan, median_miles_clocked, inplace = True)
df_car['CAR AGE'].replace(np.nan, median_car_age, inplace = True)
df_car['POSTAL CODE'].replace(np.nan, median_postal_code, inplace = True)

# Replacing the NULL values in  categorical columns using MODE


mode_sex = df_car['SEX'].mode().values[0]
mode_martial_status = df_car['MARITAL STATUS'].mode().values[0]
mode_education = df_car['EDUCATION'].mode().values[0]
mode_job = df_car['JOB'].mode().values[0]
mode_use = df_car['USE'].mode().values[0]
mode_city = df_car['CITY'].mode().values[0]
mode_car_type = df_car['CAR TYPE'].mode().values[0]


df_car['SEX']= df_car['SEX'].replace(np.nan, mode_sex)
df_car['MARITAL STATUS']= df_car['MARITAL STATUS'].replace(np.nan, mode_martial_status)
df_car['EDUCATION']= df_car['EDUCATION'].replace(np.nan, mode_education)
df_car['JOB']= df_car['JOB'].replace(np.nan, mode_job)
df_car['USE']= df_car['USE'].replace(np.nan, mode_use)
df_car['CITY']= df_car['CITY'].replace(np.nan, mode_city)
df_car['CAR TYPE']= df_car['CAR TYPE'].replace(np.nan, mode_car_type)

df_car.isnull().sum()

# Handling Duplicate Records
duplicate = df_car.duplicated()
print(duplicate.sum())
df_car[duplicate]

"""Since we have 14 duplicate records in the data, we will remove this from the data set so that we get only distinct records.

Post removing the duplicate, we will check whether the duplicates have been removed from the data set or not.
"""

# code to drop the duplicate
df_car.drop_duplicates(inplace=True)

dup = df_car.duplicated()
dup.sum()

"""**Handling Outlier**

Outliers, being the most extreme observations, may include the sample maximum or sample minimum, or both, depending on whether they are extremely high or low.

However, the sample maximum and minimum are not always outliers because they may not be unusually far from other observations.

We Generally identify outliers with the help of boxplot, so here box plot shows some of the data points outside the range of the data.
"""

df_car.boxplot(column=['INCOME'])
plt.show()

"""Looking at the box plot, it seems that the variables INCOME, have outlier present in the variables. These outliers value needs to be teated and there are several ways of treating them:

- Drop the outlier value

- Replace the outlier value using the IQR
"""

# creating a user defined function called remove_outlier for getting the threshold value

def remove_outlier(col):
  sorted(col)
  Q1, Q3 = col.quantile([0.25, 0.75])
  IQR = Q3-Q1
  lower_range = Q1-(1.5 * IQR)
  upper_range = Q3+(1.5 * IQR)
  return lower_range, upper_range

lower_income, upper_income = remove_outlier(df_car['INCOME'])

df_car['INCOME'] = np.where(df_car['INCOME'] > upper_income, upper_income, df_car['INCOME'])

df_car['INCOME'] = np.where(df_car['INCOME'] < lower_income, lower_income, df_car['INCOME'])

"""After removing outlier, let us check it with boxplot"""

df_car.boxplot(column=['INCOME'])
plt.show()

"""**Bivariate Analysis**

When we talk about bivariate analysis, it means **analyzing 2 variables**. Since we know there are numerical and categorical variables, there is a way of analyzing these variables as shown below:

**Numerical vs. Numerical**
1. Scatterplot
2. Line plot
3. Heatmap for correlation
4. Joint plot

**Categorical vs. Numerical**
1. Bar chart
2. Violin plot
3. Categorical box plot
4.Swarm plot

Two Categorical Variables
1. Bar chart
2. Grouped bar chart
3. Point plot
"""

# if we need to find the correlation
df_car.corr()

"""**Normalizing and Scaling**

Often the variables of the data set **are of different scales i.e. one variable is in millions and others in only 100.**

For e.g. in our data set Income is having values in thousands and age in just two digits. Since the data in these variables are of different scales, it is tough to compare these variables.

Feature scaling (also known as data normalization) is the method used to standardize the range of features of data.

Since the range of values of data may vary widely, it becomes a necessary step in data preprocessing while using machine learning algorithms.

In this method, we convert variables with different scales of measurements into a single scale.

StandardScaler normalizes the data using the formula (x-mean)/standard deviation.

So we will be doing this only for the numerical variables.
"""

# we use sklearn preprocessing using the function Standard Scaler

from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale

df_car['INCOME'] = std_scale.fit_transform(df_car[['INCOME']])
df_car['TRAVEL TIME'] = std_scale.fit_transform(df_car[['TRAVEL TIME']])
df_car['CAR AGE'] = std_scale.fit_transform(df_car[['CAR AGE']])
df_car['POSTAL CODE'] = std_scale.fit_transform(df_car[['POSTAL CODE']])
df_car['MILES CLOCKED'] = std_scale.fit_transform(df_car[['MILES CLOCKED']])

df_car.head()

"""**ENCODING**

One-Hot-Encoding is used to create dummy variables to replace the categories in a categorical variable into features of each category and represent it using 1 or 0 based on the presence or absence of the categorical value in the record.

This is required to do since the machine learning algorithms only work on the numerical data.

That is why there is a need to convert the categorical column into a numerical one.

**get_dummies is the method** that creates a dummy variable for each categorical variable.
"""

dummies = pd.get_dummies(df_car[['MARITAL STATUS', 'SEX', 'EDUCATION', 'JOB', 'USE', 'CAR TYPE', 'CITY']],
                         columns = ['MARITAL STATUS', 'SEX', 'EDUCATION', 'JOB', 'USE', 'CAR TYPE', 'CITY'],
                         prefix = ['married', 'sex', 'education', 'job', 'use', 'cartype', 'ciyt'], drop_first= True).head()

dummies.head()

columns = ["MARITAL STATUS", "SEX", "EDUCATION","JOB","USE", "CAR TYPE", "CITY"]
df_car = pd.concat([df_car, dummies], axis=1)

# drop original column 'fuel type' from df

df_car.drop(columns, axis= 1, inplace=True )

df_car.head()

