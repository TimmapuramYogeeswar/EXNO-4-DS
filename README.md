# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()      
```
![image](https://github.com/user-attachments/assets/a815a5dd-8b16-4fcf-9fc1-fac9a64d22c1)
```
df_null_sum=df.isnull().sum()
df_null_sum    
```
![image](https://github.com/user-attachments/assets/e9404cbc-de6a-46c6-a2b4-b0bdf6998782)
```
df.dropna()    
```
![image](https://github.com/user-attachments/assets/9d2bafd4-d0ca-460c-ab92-7371e32d76c5)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals    
```
![image](https://github.com/user-attachments/assets/3148b7fc-2de3-431f-8258-2d18d336d3fc)
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()    
```
![image](https://github.com/user-attachments/assets/316d5fc6-6c31-4d53-831a-4400c80ce5a0)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10))    
```
![image](https://github.com/user-attachments/assets/ddf93c45-be74-4a6a-9b25-ede77dae4053)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)    
```
![image](https://github.com/user-attachments/assets/b3d0dc72-1d1f-4359-ae5f-ebdfc8f74b30)
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()    
```
![image](https://github.com/user-attachments/assets/080fcbe2-1519-4503-8c8a-559990d6fad7)
```
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3  
```
![image](https://github.com/user-attachments/assets/6f7344f0-c74a-45cd-91fc-b1a94a69fb30)
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head() 
```
![image](https://github.com/user-attachments/assets/c161feae-639e-4730-b652-901e93417faf)
```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()) 
```
![image](https://github.com/user-attachments/assets/9817073d-64ba-45f5-9292-8dceb3cfdc8e)
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info() 
```
![image](https://github.com/user-attachments/assets/1983c2b0-5743-4ec1-a161-b342bd4b7d41)
```
df.info()
```
![image](https://github.com/user-attachments/assets/eccc08f1-2181-4712-a6f9-31a8294ccabc)
```
df_null_sum=df.isnull().sum()
df_null_sum!
```
[image](https://github.com/user-attachments/assets/ef7c0d81-0c56-4a7d-ab11-4b106e02deaf)
```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/184aaaf4-0e84-46f2-9393-41f5971caa2b)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/fca49a3b-455a-4e7e-bb6e-7ce9fa3cd6fd)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/55c7e99d-6502-40c9-aab5-ff5c4c78138a)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/4915fb18-d7fe-497a-9769-56b5c2bd83f4)
```
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/921d9b46-45df-4f86-84a4-15e2d0ce0b8f)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/2f28ac1d-d87b-4c01-9120-48525c988327)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/57846789-fa9a-4cbf-96ca-c6fe4f729749)
```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/1763f96c-7bdb-4d90-b5b5-9c7bcac33f7a)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/7f2338c9-16a3-4ebb-becf-4db1161f74f6)
```
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/48cd9b84-68cc-4423-b8aa-887641aaa2a5)
```
# @title
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# @title
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/36b10d7b-1e15-4204-9b12-2a5ab2c9dc12)
```
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/c7a119a0-22b6-44c9-898c-e57fe2c2bc5a)
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/bfcdc868-497d-4a48-bf65-11cb91ae0564)
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/f3e460a0-ce05-4418-acbc-f96305daf7ab)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/1e700fcc-6452-45ba-a87d-a9991b9dc44c)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/6dc49779-3acd-4458-be86-ecc2178cc7c2)
```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/5a87e93b-41d4-47a0-9b56-fbac99e7f199)
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/f036a7c1-0849-42f3-b8ed-73687dcb9d7b)

# RESULT:
    Thus,The given data is read and performed Feature Scaling and Feature Selection process and saved the data to a file.
