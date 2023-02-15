import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA #prinicipal component Analysis to decompose the data into lower dimensions to plot the data
import numpy as np
import matplotlib.pyplot as plt
st.title("ML MODEL ACCURACY CALCULATOR")
st.write("""
# streamlit Example
which one is best?
""")
dataset_name=st.sidebar.selectbox("select Dataset",("Iris","Breast Cancer","wine Data Set"))
classifier_name=st.sidebar.selectbox("select Classifier",("KNN","SVM","Random Forest"))
def get_dataset(dataset_name):
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    else:
        data=datasets.load_wine()
    X=data.data
    y=data.target
    return X,y
X,y=get_dataset(dataset_name)
st.write("shape of data",X.shape)
st.write("number of classes",len(np.unique(y)))
def add_parameter_ui(clf_name):
    params={}
    if clf_name=="KNN":
        k=st.sidebar.slider("k",1,15)
        params["k"]=k
    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        params["C"]=C
    else:
        n_estimators=st.sidebar.slider("n_estimators",1,100) #no of trees
        max_depth=st.sidebar.slider("max_depth",2,15)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators
    return params
params=add_parameter_ui(classifier_name)
def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["k"])
    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
         clf=RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(y_train.shape)


clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"accuracy={acc}")

#plotting
pca=PCA(2)
X_projected=pca.fit_transform(X)
x1=X_projected[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.colorbar()
st.set_option('deprecation.showPyplotGlobalUse', False)
# we can use plt.show() with normal code but we want result in stream lit so
st.pyplot()
