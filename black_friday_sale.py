# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 14:33:27 2018

@author: someshwar.kale
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_reading():
    test = pd.read_csv('black_friday_test.csv')
    train = pd.read_csv('black_friday_train.csv')
    return train, test

def Multivariate_analysis(train):
    matrix = train.corr()
    f, ax = plt.subplots(figsize = (5,3))
    sns.heatmap(matrix, vmax = 0.8, square = True, cmap = 'BuPu');

def Univariate_analysis(train):
    train.describe()
    train.apply(lambda x: len(x.unique()))
    missing_values_variables = []
    for variable in train.columns:
        if train[variable].isnull().sum()>0:
            missing_values_variables.append(variable)

    index_variables = ['User_ID','Product_ID','Purchase']
    missing_values_variables = missing_values_variables + index_variables
    catagorical_variables = [x for x in train.dtypes.index]
    catagorical_variables = [x for x in catagorical_variables if x not in missing_values_variables]
    numerical_variables = ['Purchase']

    plt.figure(2)
    count = 1
    for i in catagorical_variables:
        plt.subplot(5,2,count)
        train[i].value_counts(normalize=True).plot.bar(figsize=(10,20), title= '%s'%i)
        plt.grid(True)
        count = count+1
    plt.show()
    
    plt.figure(3)
    count = 1
    for variable in numerical_variables:
        plt.subplot(5,2,count)
        sns.distplot(train[variable]);
        plt.subplot(5,2,count+1)
        train[variable].plot.box(figsize=(7,7))
        plt.grid(True)
        count = count+2
        plt.show()

def Bivariate_analysis(train,test):
    train['source']='train'
    test['source']='test'
    data = pd.concat([train,test],ignore_index=True)
    data.columns
    variableList = ['Age', 'City_Category', 'Gender', 'Marital_Status', 'Occupation','Stay_In_Current_City_Years']
    plt.figure(4)
    count = 1
    for variable in variableList:
        plt.subplot(5,2,count)
        catagory_wise_purchase = data.groupby(variable)['Purchase'].sum().value_counts(normalize=True)
        catagory_wise_purchase.plot.bar(figsize=(10,20), title= '%s'%variable)
        plt.grid(True)
        count = count+1
    plt.show()
    ## we can understand that the avriable categories are indenendent and does not show any impact on purchase
    ## so better to trea them as a single category dimensionality reduction

def exploratory_analysis(train,test):
    Multivariate_analysis(train)
    Univariate_analysis(train)
    Bivariate_analysis(train,test)

def dataPreprocessing(train,test):
    ## data preproccessing 
    train['source']='train'
    test['source']='test'
    data = pd.concat([train,test],ignore_index=True)
    #data = data.drop(['Product_Category_3','Product_Category_2'], axis= 1)
    data =data.drop(data[data['Product_Category_1'] == 19].index)
    data =data.drop(data[data['Product_Category_1'] == 20].index)
    
    data = pd.get_dummies(data,columns=['City_Category'])
    data.Age.unique()
    data['Age'] = data['Age'].map({'0-17':'15', '55+':'55', '26-35':'30', '46-50':'48', '51-55':'53', '36-45':'40', '18-25':'21'})
    
    data.Stay_In_Current_City_Years.unique()
    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].map({'2':'2', '4+':'4', '3':'3', '1':'1', '0':'0'})
    
    d = {'M': 1,'F':0}
    data['Gender'] = train['Gender'].map(d)
    
    train = data.loc[data['source']=="train"]
    test = data.loc[data['source']=="test"]
    #Drop unnecessary columns:
    test.drop(['Purchase','source'],axis=1,inplace=True)
    train.drop(['source'],axis=1,inplace=True)

    return train, test

def featureExtraction(train,test):
    ## user count: number of times user made purchase
    train['source']='train'
    test['source']='test'
    data = pd.concat([train,test],ignore_index=True)
    data.User_ID.unique()
    data['user_count'] = data.User_ID.groupby(data.User_ID).transform('count')
    data['product_count'] = data.Product_ID.groupby(data.Product_ID).transform('count')
    data['product_mean'] = data['Product_ID'].map(train.Purchase.groupby(train.Product_ID).mean())
    data['highFlag'] = data['Purchase']>data['product_mean']
    d = {True: 1,False:0}
    data['highFlag'] = data['highFlag'].map(d)
    
    train = data.loc[data['source']=="train"]
    test = data.loc[data['source']=="test"]
    #Drop unnecessary columns:
    test.drop(['Purchase','source'],axis=1,inplace=True)
    train.drop(['source'],axis=1,inplace=True)
    
    return train, test

def dimensionality_reduction(train):
    X_train = train[['Gender','Age','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_ID','User_ID']]
    Y_train = train['Purchase']
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(train)
    # Applying LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components = 2)
    X_train = lda.fit_transform(X_train, Y_train)
    X_train['Purchase'] = Y_train
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components =2)
#    X_train = pca.fit_transform(X_train)
#    explained_variance = pca.explained_variance_ratio_
#    print (explained_variance)
    return X_train

def modelfit(model, train, test, predictors, target):
    model.fit(train[predictors], train[target])
    train_predictions = model.predict(train[predictors])
    #Perform cross-validation:
    from sklearn import cross_validation, metrics
    cv_score = cross_validation.cross_val_score(model, train[predictors], train[target], cv=20, scoring='mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(train[target].values, train_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    #Predict on testing data:
    test[target] = model.predict(test[predictors])
    return test

train,test = data_reading()
## exploratory analysis
#exploratory_analysis(train,test)
train, test = dataPreprocessing(train, test)
train, test = featureExtraction(train, test)
#train, test = dimensionality_reduction(train,test)
y = train.Purchase
train[['Age','Stay_In_Current_City_Years']] = train[['Age','Stay_In_Current_City_Years']].astype(float)
test[['Age','Stay_In_Current_City_Years']] = test[['Age','Stay_In_Current_City_Years']].astype(float)
X_train = train.drop(['Product_ID','User_ID','highFlag','user_count', 'product_count','product_mean'],axis =1)
X_test = test.drop(['Product_ID','User_ID','highFlag','user_count', 'product_count','product_mean'],axis =1)

#model = XGBRegressor(X_train,y,X_test,cv=5,objective="reg:linear",nrounds=500,max.depth=10,eta=0.1,colsample_bytree=0.5,seed=235,metric="rmse",importance=1)

target = 'Purchase'
predictors = [x for x in train.columns if x not in [target,'Product_ID','User_ID','highFlag','Purchase','user_count', 'product_count','product_mean']]
from xgboost import XGBRegressor
model = XGBRegressor()
test = modelfit(model, X_train, X_test, predictors, target)
coef1 = pd.Series(model.feature_importances_, predictors).sort_values(ascending=False)
coef1.plot(kind='bar', title='Feature Importances')
plt.show()

