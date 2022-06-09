# import
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics

import shap



# emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Seattle Energy", page_icon=":high_brightness:", layout="wide")


# config
st.set_page_config(page_title="Seattle Energy App", page_icon=":sun:")

# load machine learning model
with open('log_best_model.pkl' , 'rb') as f:
    model_app = pickle.load(f)

def main():

    #load data
    uploaded_data = open("data_model.csv", "r")
    df = pd.read_csv(uploaded_data)

    # define target and features
    y=df[['TotalGHGEmissions','SiteEnergyUse(kBtu)']]
    X=df.drop(['TotalGHGEmissions','SiteEnergyUse(kBtu)'],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state= 2022)

    # header with image and title
    st.image("seattle.png")
    st.title("Energy Prediction")

    # welcome info
    st.info("Welcome here !  Hop, upload your .csv file & you can predict the energy use by the building in Seattle and the GHG emissions by year")

    # show the table data
    st.dataframe(df)

    #0-GHG 
    #1-Energy
    y_pred=model_app.predict(X_test)

    # scores
    st.title("Scores")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⚡Energy use by kg/yard**")
        st.write("R2: ", metrics.r2_score(y_test.iloc[:,1], y_pred[:,1]))
        st.write("MAE: ", (mean_absolute_error(y_test.iloc[:,1], y_pred[:,1])))
        st.write("MSE: ", (mean_squared_error(y_test.iloc[:,1], y_pred[:,1])))
        st.write("RMSE: ", (mean_squared_error(y_test.iloc[:,1], y_pred[:,1], squared=False)))

    with col2:
        st.markdown("**☁ GHG emissions by year**")
        st.write("R2: ", metrics.r2_score(y_test.iloc[:,0], y_pred[:,0]))
        st.write("MAE: ", (mean_absolute_error(y_test.iloc[:,0], y_pred[:,0])))
        st.write("MSE: ", (mean_squared_error(y_test.iloc[:,0], y_pred[:,0])))
        st.write("RMSE: ", (mean_squared_error(y_test.iloc[:,0], y_pred[:,0], squared=False)))

if __name__ == '__main__':
    main()