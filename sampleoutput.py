#Importing the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import math
import os
from sklearn import svm,datasets
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

#Importing the dataset
os.chdir('C:/Users/chait/Desktop/Project1')
df = pd.read_csv('Train_data.csv')

#Data Preprocessing
df.dtypes
df.isnull().sum()
df['area code'] = df['area code'].astype(object)
str(df['area code'])
df.info()
df.drop(['phone number'],axis = 1,inplace = True)

#Dividing the numeric and categorical variables
cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)

#Saving Categoical columns
cat_data = df.select_dtypes(include = ['object'])
categorical_variables = list(cat_data.columns.values)

#Assigning levels to the categories
for i in range(0,df.shape[1]):
    if(df.iloc[:,i].dtypes == 'object'):
        df.iloc[:,i] = pd.Categorical(df.iloc[:,i])
        df.iloc[:,i] = df.iloc[:,i].cat.codes
for i in categorical_variables:
    df[i] = df[i].astype(object)

#Exploratory Data Analysis
sns.countplot(x="Churn",data=df)

#Outlier Analysis
plt.boxplot(df['account length'])
for i in numeric_variables:
    print(i)
    q75,q25 = np.percentile(df.loc[:,i],[75,25])
    iqr = q75-q25
    
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    df.loc[df.loc[:,i]<min ,i] = np.nan
    df.loc[df.loc[:,i]>max ,i] = np.nan

#Replacing the outliers with median value    
df = df.fillna(df.median(),inplace = True)

for i in categorical_variables:
    df[i] = df[i].astype(object)

cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)

#Feature Selection
f,ax = plt.subplots(figsize =(7,5))

#Generate correlation matrix
corr = cnames.corr()

#Plot using seaborne library
sns.heatmap(corr, mask =np.zeros_like(corr, dtype = np.bool), 
            cmap =sns.diverging_palette(220, 10, as_cmap =True),
            square = True, ax = ax)
df.drop(['total day minutes','total eve minutes','total night minutes'],axis = 1,
        inplace = True)

cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)

#Feature scaling/Normalization
%matplotlib inline
for i in numeric_variables:
    plt.hist(df[i], bins ='auto')
    plt.show

#Normalization of the data
for i in numeric_variables:
    print(i)
    a = df[i].max()
    b = df[i].min()
    df[i] = (df[i]- b)/(a-b)
#Chi-sqare test
categorical_variables = categorical_variables[:-1] 
for i in categorical_variables:
    print(i)
    chi2, p, dof, ex =chi2_contingency(pd.crosstab(df.Churn,df[i]))
    print(p)
df.drop(['area code','voice mail plan'],axis = 1, inplace =True)
X_train = df.iloc[:,0:14]
Y_train = df.iloc[:,14]
Y_train = Y_train.astype('int') 

#########Importing the test dataset#########

df = pd.read_csv('Test_data.csv')

#Preparing the sample input data
df = df.sample(n = 500)

#Data Preprocessing
df.isnull().sum()
#df['area code'] = df['area code'].astype(object)
df.drop(['phone number'],axis = 1,inplace = True)

#Dividing the numeric and categorical variables
cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)

#Saving Categoical columns
cat_data = df.select_dtypes(include = ['object'])
categorical_variables = list(cat_data.columns.values)

#Assigning levels to the categories
for i in range(0,df.shape[1]):
    if(df.iloc[:,i].dtypes == 'object'):
        df.iloc[:,i] = pd.Categorical(df.iloc[:,i])
        df.iloc[:,i] = df.iloc[:,i].cat.codes
for i in categorical_variables:
    df[i] = df[i].astype(object)

#Feature Selection
df.drop(['total day minutes','total eve minutes','total night minutes'],axis = 1,
        inplace = True)
df.drop(['area code','voice mail plan'],axis = 1, inplace =True)

cnames = df.select_dtypes(exclude = ['object'])
numeric_variables = list(cnames.columns.values)

#Saving Categoical columns
cat_data = df.select_dtypes(include = ['object'])
categorical_variables = list(cat_data.columns.values)

#Normalization of the data
for i in numeric_variables:
    print(i)
    a = df[i].max()
    b = df[i].min()
    df[i] = (df[i]- b)/(a-b)

X_test = df.iloc[:,0:14]
Y_test = df.iloc[:,14]

#Converting the target variable to int type for
Y_test = Y_test.astype('int') 
Y_train = Y_train.astype('int')
 
#Decision Tree
clf = tree.DecisionTreeClassifier(criterion ='entropy').fit(X_train,Y_train)

#Predict
Y_pred = clf.predict(X_test)
fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred)

#Check accuracy of the model
print ('Accuracy:', accuracy_score(Y_test, Y_pred)*100)
print ('F1 score:', f1_score(Y_test, Y_pred))
print ('Sensitivity or Recall:', recall_score(Y_test, Y_pred))
print ('Precision:', precision_score(Y_test, Y_pred))
print ('\n clasification report:\n', classification_report(Y_test, Y_pred))
print ('\n confussion matrix:\n',confusion_matrix(Y_test, Y_pred))

#ROC PLot
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(Y_test, Y_pred)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

df.to_csv('sampleinput_python.csv')
np.savetxt("output_python.csv", Y_pred, newline=" ")
########################END###########################