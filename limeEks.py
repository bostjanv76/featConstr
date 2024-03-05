# Load useful libraries
import sys
import os
from datetime import datetime
import time
import pandas as pd
import xgboost as xgb
from typing import Union, Any
# Import the LimeTabularExplainer module
from lime.lime_tabular import LimeTabularExplainer
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def substring_before_dot(string):
    dot_index = string.find('.')
    if dot_index != -1:
        return string[:dot_index]
    else:
        return string
 
def write_to_file(filename, text):
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            file.write(text)
            #print("New file created: {}".format(filename))
    else:
        with open(filename, 'a') as file:
            file.write(text)
            #print("Text appended to file: {}".format(filename)) 
# Separate Features and Target Variables
ekspl_data: Union[DataFrame, Any] = pd.read_csv('Lime/explainData.csv')
demo_data: Union[DataFrame, Any] = pd.read_csv('Lime/trainFold.csv')
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\cLogicalConcBEasy.csv')   #binary class
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\Glass.csv')   #multi class problem - UCI
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\cMultiVClassDisAttr.csv')    #multi value (3) class
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\cConcept.csv')    #multi value (3) class
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\cBinClassNumDisAttr.csv')
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\Australian credit.csv')
#demo_data: Union[DataFrame, Any] = pd.read_csv('D:\python_scripts\columnar-test.csv')

classToExplain = int(sys.argv[1])
alg = (sys.argv[2]).lower()
dataset_name = sys.argv[3]
class_values = sys.argv[4]
new_tuple = (classToExplain, )

# last column is class column
last_column = demo_data.columns[-1]
class_values_list=class_values.split(',')

# Label Encoding features 
categorical_feat =list(demo_data.select_dtypes(include=["object"]))

# Using label encoder to transform string categories to integer labels
# LimeTabularExplainer is assuming that the input is an array of floats, and that categorical features have been encoded 
le = LabelEncoder()
for feat in categorical_feat:
    demo_data[feat] = le.fit_transform(demo_data[feat]).astype('int')


X = demo_data.drop(columns=last_column)
y = demo_data[last_column]

X = X.values
y = y.values

categorical_feat =list(ekspl_data.select_dtypes(include=["object"]))#it should be the same as in demo_data
le = LabelEncoder()
for feat in categorical_feat:
    ekspl_data[feat] = le.fit_transform(ekspl_data[feat]).astype('int')

E = ekspl_data.drop(columns=last_column)
e = ekspl_data[last_column]

E = E.values
e = e.values

xgbOrRf = True
if (alg !='xgb'):
    xgbOrRf = False     #model to explain XGB or RF

# Build the model
rf_clf = RandomForestClassifier(n_estimators=100) #max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt” -> max_features=sqrt(n_features)

rf_clf.fit(X, y)


# Train a model using the scikit-learn API
xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.3, max_depth=3, gamma=1)
xgb_classifier.fit(X, y)

Inst_expl=E
# Get the class names
class_names = class_values_list

# Fit the Explainer on the training data set using the LimeTabularExplainer
explainer = LimeTabularExplainer(X,
                                 class_names = class_names,
                                 mode = 'classification')

total_execution_time = 0
current_datetime = datetime.now()
file_name = 'Lime/execTimes.log'

for x in Inst_expl:
    if xgbOrRf:
        start = time.time()
        ttt = explainer.explain_instance(x, xgb_classifier.predict_proba, labels=new_tuple, num_samples=5000)
        end = time.time()
        time_taken = end - start
        total_execution_time += time_taken
    else:
        start = time.time()
        ttt=explainer.explain_instance(x, rf_clf.predict_proba, labels=new_tuple, num_samples=5000)
        end = time.time()
        time_taken = end - start
        total_execution_time += time_taken
    listOfTuples=list(ttt.local_exp.values())
    sorted_list=sorted(listOfTuples[0])
    clean_expl = [tup[1] for tup in sorted_list]
    res = str(clean_expl)[1:-1].replace(", ", ',')
    print(res)

text='Total explanation time: '+str(total_execution_time)+' [s] for dataset: '+dataset_name+' time of entry: '+str(current_datetime)+'\n'
write_to_file(file_name, text)