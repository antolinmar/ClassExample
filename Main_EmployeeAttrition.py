# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:31:52 2024

@author: A. Martínez-Martínez

Problem statement and dataset provided by:
https://www.youtube.com/watch?v=n2JACagUziA&ab_channel=TheDataFutureLab

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

savepath=r".\images\\"
saveFLAG= True

filepath=r'.\dataset'
filename=r'\Dataset01-Employee_Attrition.csv'

df = pd.read_csv(filepath+filename)

print(df.head()) 

#%% Remove duplicate rows

df_1=df.drop_duplicates()

#%% Check for missing values

df_1.isnull().sum()

#%% EDA
# In this section, we perform an Exploratory Data Analysis to see the big picture of the prblema nd infere interesting ideas from the dataset.
# This step is useful to understand the problem and detect potential error in the model prediction.

#%% Check kind of data in our pandas dataframe

df_1.dtypes

df_1.head

# Obtain variables with numerical data:
df_1._get_numeric_data()
    
#Obtain varibales with categorical data
df.select_dtypes(include=['object'])


#%% How many employees left the company?

plt.figure()
plt.pie([sum(df_1['left']==0),sum(df_1['left']==1)],labels=('Remained','Left'))
plt.title('Employees status')

#%% How are they distributed by departments?

species = df_1['Department'].unique()

count_left_spec=[]
count_remain_spec=[]
for idept in species:
    
    
    count_left_spec.append(df_1.loc[(df_1['left']==1) & (df_1['Department']== idept)].shape[0])
    count_remain_spec.append(df_1.loc[(df_1['left']==0) & (df_1['Department']== idept)].shape[0])

weight_counts = {
    "Left": np.array(count_left_spec),
    "Remain": np.array(count_remain_spec),
}
width = 0.5

fig, ax = plt.subplots()
bottom = np.zeros(len(species))

for boolean, weight_count in weight_counts.items():
    p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom)
    bottom += weight_count

ax.set_title("Departments distributions")
ax.legend(loc="upper right")
plt.xticks(rotation='vertical')

plt.tight_layout()

if saveFLAG:
    plt.gcf().savefig(savepath + 'DepartmentDist_1.png')
    
plt.close()    
# Another plot option (easier):
    
pd.crosstab(df_1.Department,df_1.left).plot(kind='bar')

if saveFLAG:
    plt.gcf().savefig(savepath + 'DepartmentDist_2.png')
    
plt.close()    


#%%  What is the distribution of monthly hours of 'left' employees?

data1 = df_1.loc[df_1['left'] == 1, 'average_montly_hours'].to_numpy()
data2 = df_1.loc[df_1['left'] == 0, 'average_montly_hours'].to_numpy()

# Combine the data into a list
combined_data = [data1, data2]

# Create a figure and axis object
fig, ax = plt.subplots()

# Create boxplots for both data sets
ax.boxplot(combined_data)

# Customize labels
ax.set_xticklabels(['Left', 'Not Left'])
ax.set_title('Average montly hours')
# Show the plot
plt.tight_layout()

if saveFLAG:
    plt.gcf().savefig(savepath + 'MonthlyHours.png')
    
plt.close()    

#%% What is the distribution of last rate of 'left' employees?

data1 = df_1.loc[df_1['left'] == 1, 'last_evaluation'].to_numpy()
data2 = df_1.loc[df_1['left'] == 0, 'last_evaluation'].to_numpy()

# Combine the data into a list
combined_data = [data1, data2]

# Create a figure and axis object
fig, ax = plt.subplots()

# Create boxplots for both data sets
ax.boxplot(combined_data)

# Customize labels
ax.set_xticklabels(['Left', 'Not Left'])
ax.set_title('Last evaluation')

# Show the plot
plt.tight_layout()

if saveFLAG:
    plt.gcf().savefig(savepath + 'LastRate.png')
    
plt.close()    

#%% What is the satisfaction level of the 'left' employees?

data1 = df_1.loc[df_1['left'] == 1, 'satisfaction_level'].to_numpy()
data2 = df_1.loc[df_1['left'] == 0, 'satisfaction_level'].to_numpy()

# Combine the data into a list
combined_data = [data1, data2]

# Create a figure and axis object
fig, ax = plt.subplots()

# Create boxplots for both data sets
ax.boxplot(combined_data)

# Customize labels
ax.set_xticklabels(['Left', 'Not Left'])
ax.set_title('Satisfaction Level')

# Show the plot
plt.tight_layout()

if saveFLAG:
    plt.gcf().savefig(savepath + 'SatisfactionLeel.png')
    
plt.close()    

#%% Percentage of work accident by 'left' employee

data1 = df_1.loc[df_1['left'] == 1, 'Work_accident'].to_numpy()
data2 = df_1.loc[df_1['left'] == 0, 'Work_accident'].to_numpy()

print('Percentage of employees who left and sufferend a work accident:' + str(data1.sum()/data1.shape*100))

print('Percentage of employees who remained and sufferend a work accident:' + str(data2.sum()/data2.shape*100))


#%% Crosstabs to relate our classification and a categorical value

pd.crosstab(df_1.salary,df_1.left).plot(kind='bar')

if saveFLAG:
    plt.gcf().savefig(savepath + 'SalaryDist.png')
    
plt.close()    


#%% Feature engineering
#%% Label encoding: Categorical into numerical

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df_1['salary'] = label_encoder.fit_transform(df_1['salary'])
df_1['Department'] = label_encoder.fit_transform(df_1['Department'])

#%% Define dependent and independent variables for the model:

X = df_1.drop('left',axis=1) #Input data
Y = df_1['left'] #Output data

#%% Split the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=23)

#%% Scale the data
# Each dataset is scaled independently, as it would be done in a real case, were the 'test' data is provided independently from the data used to fit the model.
# If we do not consider this scenario, the scaling strategy may change.

from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

xtrain_scaled = std_scaler.fit_transform(x_train)

xtest_scaled = std_scaler.fit_transform(x_test)

#%% Model implementation Num. 1: Random forest

#% Random forest instance

from sklearn.ensemble import RandomForestClassifier

Random_forest_model = RandomForestClassifier()

# Model training

Random_forest_model.fit(xtrain_scaled,y_train)

# Model test

y_pred=Random_forest_model.predict(xtest_scaled)

#%%  Model evaluation
# Confusion Matrix

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
plt.figure()
sns.heatmap(cm, annot = True, fmt ='d')

if saveFLAG:
    plt.gcf().savefig(savepath + 'ConfusionMatrix.png')
    
plt.close()    


# Accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true

from sklearn.metrics  import accuracy_score
model_accuracy = accuracy_score(y_test, y_pred)

# Precision

from sklearn.metrics  import precision_score
model_precision = precision_score(y_test, y_pred)

# Recall

from sklearn.metrics  import recall_score
model_recall = recall_score(y_test, y_pred)

# F1-score

from sklearn.metrics  import f1_score
model_f1 = f1_score(y_test, y_pred)

# Classification report 

from sklearn.metrics  import classification_report
print(classification_report(y_test, y_pred))

# ROC Curve
from sklearn import metrics
metrics.plot_roc_curve(Random_forest_model, xtest_scaled, y_test) 

# Feature importance

score_list = Random_forest_model.feature_importances_

list_of_features = list(X.columns)

score_df = pd.DataFrame({"Feature": list_of_features,'Score': score_list})

score_df.sort_values(by = 'Score', ascending = False)

plt.figure()
plt.barh(range(len(list_of_features)), Random_forest_model.feature_importances_)
plt.yticks(np.arange(len(list_of_features)),list_of_features)
plt.tight_layout()

if saveFLAG:
    plt.gcf().savefig(savepath + 'FeaturesImportanceModel1.png')
    
plt.close()    


#%% k-fold cross validation for Random Forest Model
# We are going to apply the k-fold strategy to create 5 different random forect classifiers and select the best one using as metric the accuracy.
from sklearn.model_selection import cross_val_score

scores= cross_val_score(Random_forest_model, xtrain_scaled, y_train, cv = 5, scoring='accuracy')

print(scores)

#%% Model implementation Num. 2: XGBoost
import xgboost as xgb

data_train_mat = xgb.DMatrix(xtrain_scaled, label=y_train)
data_test_mat = xgb.DMatrix(xtest_scaled, label=y_test)

XG_parameters = {"booster":"gbtree", "max_depth": 2, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
rounds = 10

evaluation = [(data_test_mat, "eval"), (data_train_mat, "train")]

xgboost_model = xgb.train(XG_parameters, data_train_mat, rounds, evaluation)


#%% Compare models using ROc curve plot and AUC

from sklearn import metrics
def buildROC(target_test,test_preds,titletext):
    fpr, tpr, threshold = metrics.roc_curve(target_test, test_preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.title(titletext)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.gcf().savefig(savepath + titletext + '.png')
    # plt.close()
    
#XGBoost-based model    
test_preds_XG=xgboost_model.predict(data_test_mat) 

buildROC(y_test.values,test_preds_XG, 'XGBoost Results')

if saveFLAG:
    plt.gcf().savefig(savepath + 'XGBoost_ROC.png')
    
plt.close()    

#Random forest model
test_preds_randforest=Random_forest_model.predict(xtest_scaled)

buildROC(y_test.values,test_preds_randforest,'Random Forest Results')

if saveFLAG:
    plt.gcf().savefig(savepath + 'RandomForest_ROC.png')
    
plt.close()    