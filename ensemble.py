import numpy as np 

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import random

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor
 

import xgboost as xgb

from sklearn.metrics import roc_curve, auc

def read_data(run_num, prob):

    normalise = False
    
    if prob == 'classifification': #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv
        data_in = genfromtxt("data/pima.csv", delimiter=",")
        data_inputx = data_in[:,0:8] # all features 0, 1, 2, 3, 4, 5, 6, 7 
        data_inputy = data_in[:,-1] # this is target - so that last col is selected from data

    elif prob == 'regression': # energy - regression prob: https://archive.ics.uci.edu/dataset/242/energy+efficiency
        data_in = genfromtxt('data/ENB2012_data.csv', delimiter=",")  
        data_inputx = data_in[:,0:8] # all features 0, - 7
        data_inputy = data_in[:,8] # this is target - just the heating load selected from data
  

    if normalise == True:
        transformer = Normalizer().fit(data_inputx)  # fit does nothing.
        data_inputx = transformer.transform(data_inputx)
 

 
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test

 
    
def scipy_models(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num, problem):

    print(run_num, ' is our exp run')

    tree_depth = 2
 
    if problem == 'classifification':
        if type_model ==0: #SGD 
            model = MLPClassifier(hidden_layer_sizes=(hidden,), random_state=run_num, max_iter=100,solver='sgd',  learning_rate_init=learn_rate )  
        elif type_model ==1: #https://scikit-learn.org/stable/modules/tree.html  (see how tree can be visualised)
            model = DecisionTreeClassifier(random_state=0, max_depth=tree_depth) 
        elif type_model ==2:
            model = RandomForestClassifier(n_estimators=100, max_depth=tree_depth, random_state=run_num)
            
        elif type_model ==3:
            model = AdaBoostClassifier(n_estimators=100,  random_state=run_num)

        elif type_model ==4:
            model = GradientBoostingClassifier(n_estimators=10,  random_state=run_num)


    elif problem == 'regression':
        if type_model ==0: #SGD  
            model = MLPRegressor(hidden_layer_sizes=(hidden*3,), random_state=run_num, max_iter=500, solver='adam',learning_rate_init=learn_rate) 
        elif type_model ==1:  
            model = DecisionTreeRegressor(random_state=0, max_depth=tree_depth)
        elif type_model ==2: 
            model = RandomForestRegressor(n_estimators=100, max_depth=tree_depth, random_state=run_num)
        elif type_model ==3: 
            model = AdaBoostRegressor(n_estimators=100, random_state=run_num)
        elif type_model ==4:
            model = GradientBoostingRegressor(n_estimators=10,  random_state=run_num)

            
   
    # Train the model using the training sets

    model.fit(x_train, y_train)
   

    if type_model ==1:
        r = export_text(model)
        print(r)


    # Make predictions using the testing set
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    if problem == 'regression':
        perf_test =  np.sqrt(mean_squared_error(y_test, y_pred_test)) 
        perf_train=  np.sqrt(mean_squared_error(y_train, y_pred_train)) 

    if problem == 'classifification': 
        perf_test = accuracy_score(y_pred_test, y_test) 
        perf_train = accuracy_score(y_pred_train, y_train) 
        cm = confusion_matrix(y_pred_test, y_test) 
        #print(cm, 'is confusion matrix')
        #auc = roc_auc_score(y_pred, y_test, average=None) 

    return perf_test #,perf_train



def xgboost_models(x_train, x_test, y_train, y_test, type_model, hidden, learn_rate, run_num, problem):

    print(run_num, ' is our exp run')

    tree_depth = 2
 
    if problem == 'classifification':
        if type_model ==0:  
            model = xgb.XGBClassifier(colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 5, n_estimators = 100)

            

    elif problem == 'regression':
        if type_model ==0: #SGD  
            model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 5, n_estimators = 100)

            
   
    # Train the model using the training sets

    model.fit(x_train, y_train)
   

    if type_model ==1:
        r = export_text(model)
        print(r)


    # Make predictions using the testing set
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train) 

    if problem == 'regression':
        perf_test =  np.sqrt(mean_squared_error(y_test, y_pred_test)) 
        perf_train=  np.sqrt(mean_squared_error(y_train, y_pred_train)) 

    if problem == 'classifification': 
        perf_test = accuracy_score(y_pred_test, y_test) 
        perf_train = accuracy_score(y_pred_train, y_train) 
        cm = confusion_matrix(y_pred_test, y_test) 
        #print(cm, 'is confusion matrix')
        #auc = roc_auc_score(y_pred, y_test, average=None) 

    return perf_test #,perf_train


def main(): 

    max_expruns = 5

    SGD_all = np.zeros(max_expruns) 
    forest_all = np.zeros(max_expruns) 
    tree_all = np.zeros(max_expruns) 
    adaboost_all = np.zeros(max_expruns)  

    xg_all = np.zeros(max_expruns)  

    gb_all = np.zeros(max_expruns)  
 
    learn_rate = 0.01
    hidden = 8

    prob = 'classifification' #  classification  or regression 
    #prob = 'regression' #  classification  or regression 


    # classifcation accurary is reported for classification and RMSE for regression

    print(prob, ' is our problem')

 
 
    for run_num in range(0,max_expruns): 

        x_train, x_test, y_train, y_test = read_data(run_num, prob)   
        
        acc_sgd = scipy_models(x_train, x_test, y_train, y_test, 0, hidden, learn_rate, run_num, prob) #SGD 
        acc_tree = scipy_models(x_train, x_test, y_train, y_test, 1, hidden, learn_rate,  run_num, prob) #Decision Tree
        acc_forest = scipy_models(x_train, x_test, y_train, y_test, 2, hidden, learn_rate,  run_num, prob) #Random Forests
        acc_adaboost = scipy_models(x_train, x_test, y_train, y_test, 3, hidden, learn_rate,  run_num, prob) #adaboost
        acc_gb = scipy_models(x_train, x_test, y_train, y_test, 4, hidden, learn_rate,  run_num, prob) #gboost

        acc_xg = xgboost_models(x_train, x_test, y_train, y_test, 0, hidden, learn_rate,  run_num, prob) #adaboost
        

       
        SGD_all[run_num] = acc_sgd 
        tree_all[run_num] = acc_tree
        forest_all[run_num] = acc_forest
        adaboost_all[run_num] = acc_adaboost
        gb_all[run_num] = acc_gb

        xg_all[run_num] = acc_xg

    print(SGD_all,' nn_all')
    print(np.mean(SGD_all), ' mean nn_all')
    print(np.std(SGD_all), ' std nn_all')
 
    print(tree_all,  ' tree_all')
    print(np.mean(tree_all),  ' tree _all')
    print(np.std(tree_all),  ' tree _all')

    print(forest_all, hidden,' forest_all')
    print(np.mean(forest_all),  ' forest _all')
    print(np.std(forest_all),  ' forest _all')

    print(adaboost_all,  'adaboost_all')
    print(np.mean(adaboost_all),  ' adaboost _all')
    print(np.std(adaboost_all),  ' adaboost_all')

       
    print(gb_all,  'gb_all')
    print(np.mean(gb_all),  ' gb _all')
    print(np.std(gb_all),  ' gb_all')

 
    print(xg_all,  'xg_all')
    print(np.mean(xg_all),  ' xg _all')
    print(np.std(xg_all),  ' xg_all')



if __name__ == '__main__':
     main() 
