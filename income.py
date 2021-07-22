import streamlit as st 
from pycaret.regression import load_model, predict_model
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

st.title('Income classifier')

df_original = pd.read_csv("adult.csv")
df = pd.read_csv("final_adult.csv")

def get_dataset(df):
    X = df.copy()
    X.drop(["Income"],axis = 1,inplace = True)
    y = df_original[["Income"]]
    return X, y


X_original, y_original = get_dataset(df_original)
X, y =get_dataset(df)

st.write("Original Dataset")
st.write(df_original.head())



st.write("Dataset After EDA")
st.write(df)

from pycaret import classification
# setup the environment 
st.write(classification_setup = classification.setup(data= df, target='Income'))
st.write(classification.compare_models())