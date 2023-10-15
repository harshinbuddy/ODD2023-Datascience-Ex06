## EX-06 FEATURE TRANSFORMATION
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.
### EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### ALGORITHM:
-  Step1: Read the given Data.
-  Step2: Clean the Data Set using Data Cleaning Process.
-  Step3: Apply Feature Transformation techniques to all the features of the data set.
-  Step4: Print the transformed features.
## PROGRAM:
```
Developed by : Dhanesh M
Register No : 212220220009
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## OUTPUT:
### Original Data:
![ds 6 1](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/c907ca3f-0eac-4358-8e98-5a1403e746db)
### Data information:
![ds 6 2](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/4df96442-db8c-4382-ac59-ce70321c70c5)
### Data Describe:
![ds 6 3](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/619cb5c9-e037-4dd9-916a-1481d94c5c8b)
### Before transformation:
![ds 6 4](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/5d426958-3669-42af-8c0f-8e82d5812593)
![ds 6 5](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/398aecc8-565f-4587-82ba-d78a78d978dc)
![ds 6 6](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/0076352e-ae3e-466d-a36e-8c1fad0d7114)
![ds 6 7](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/32a13076-2c92-4fa3-9fd5-99849e4b91e0)
### Log transformation:
![ds 6 8](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/94f0822e-1b87-4c54-9ffa-ec8c3d92cef7)
### Reciprocal transformation:
![ds 6 9](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/a02242b5-8752-4828-a1d8-4df2e57196e9)
### Square root transformation:
![ds 6 10](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/b0fd9bfe-48b2-416c-8d3f-6e02d5eaf3e2)
![ds 6 new](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/6651776d-79ef-47a5-9d44-a8a1df250a09)
### Power transformation:
![ds 6 12 1](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/e80598b8-156c-4355-be9f-e48818e6a895)
![ds 6 12](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/6568f818-5b8c-4c68-8c73-dfbb5e68eff4)
### Quantile transformation:
![ds 6 13](https://github.com/deepikasrinivasans/ODD2023-Datascience-Ex06/assets/119393935/286770cd-194f-4144-93cf-dc255c39ab84)
## RESULT:
Thus feature transformation is done for the given dataset.

