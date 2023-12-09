import os

import numpy as np
import pandas as pd

from PIL import Image

import plotly.express as px

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,LabelEncoder

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn import metrics

import streamlit as st

icon = Image.open("logo.png")

dsaLogo = Image.open("DSA_logo.png")

dataScientistImage = Image.open("homepage.jpg")

st.set_page_config(layout='wide',page_title='Data Science Academy',page_icon=icon)

st.header('Week 8 Python Case Study')

st.sidebar.image(image=dsaLogo)

menu = st.sidebar.selectbox("",["Homepage","EDA","Modeling"])

def outlier_treatment(data_c):

    sorted(data_c)

    Q1,Q3 = np.percentile(data_c,[25,75])

    IQR = Q3-Q1

    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+(1.5*IQR)

    return lower_range,upper_range

def description_panel(df):

    st.dataframe(df)

    st.subheader("Statistical Description")
    df.describe().T

    st.subheader("Balance of Data")
    st.bar_chart(df.iloc[:,-1].value_counts())

    null_df = df.isnull().sum().to_frame().reset_index()
    null_df.columns = ["Columns","Counts"]

    p1,p2,p3 = st.columns([2,1,2])

    p1.subheader("Null Variables")
    p1.dataframe(null_df)

    p2.subheader("Imputation")
    cat_m = p2.radio("Categorical",["Mode","BackFill","ForwardFill"])
    num_m = p2.radio("Numerical",["Mode","Median"])

    p2.subheader("Feature Engineering")
    balance_problem = p2.checkbox("Over Sampling")
    outlier_problem = p2.checkbox("Clean Outlier")

    if p2.button("Data preprocessing"):

        cat_cols = df.iloc[:,:-1].select_dtypes(include="object").columns
        num_cols = df.iloc[:,:-1].select_dtypes(exclude="object").columns

        if cat_cols.size > 0:

            if cat_m == "Mode":
                imp_cat = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

            elif cat == "BackFill":
                df[cat_cols].fillna(method="backfill",inplace=True)

            elif cat == "ForwardFill":
                df[cat_cols].fillna(method="ffill",inplace=True)

        if num_cols.size>0:

            if num_m == "Mode":
                imp_num = SimpleImputer(missing_values=np.nan,strategy="most_frequent")
            elif num_m == "Median":
                imp_num = SimpleImputer(missing_values=np.nan,strategy="median")

            df[num_cols] = imp_num.fit_transform(df[num_cols])

        df.dropna(axis=0,inplace=True)

        if balance_problem:
            over_sample = RandomOverSampler()
            X = df.iloc[:,:-1]
            y = df.iloc[:,[-1]]

            X,y = over_sample.fit_resample(X,y)

            df = pd.concat([X,y],axis=1)

        if outlier_problem:

            for col in num_cols:
                lower_bound,upper_bound = outlier_treatment(df[col])
                df[col] = np.clip(df[col],a_min=lower_bound,a_max=upper_bound)

        null_df = df.isnull().sum().to_frame().reset_index()
        null_df.columns = ["Columns","Counts"]

        p3.subheader("Null Variables")
        p3.dataframe(null_df)

        st.subheader("Balance of Data")
        st.bar_chart(df.iloc[:,-1].value_counts())

        heatmap = px.imshow(df.select_dtypes(exclude="object").corr())
        st.plotly_chart(heatmap)
        st.dataframe(df)

        if os.path.exists("model.csv"):
            os.remove("model.csv")

        df.to_csv("model.csv",index=False)

if menu == "Homepage":

    st.header("Homepage")
    st.image(dataScientistImage,use_column_width="always")

    dataset = st.selectbox("Select dataset",["Loan Prediction","Water Potability"])

    st.markdown("Selected: **{0}** Dataset".format(dataset))

    if dataset == "Loan Prediction":

        st.warning("You selected **Loan Prediction** dataset")

        st.info("""
            **Loan_ID** - Unique Loan ID\n
            **Gender** - Male or Female\n
            **Married** - Marital Status (Y/N)\n
            **Dependents** - Number of Dependents\n
            **Education** - Applicant Education (Graduate/Under Graduate)\n
            **Self_Employed** - Self Employment Status (Y/N)\n
            **ApplicantIncome** - Appliant Income Amount\n
            **CoapplicantIncome** - Coapplicant Income Amount\n
            **LoanAmount** - Loan Amount in Thousands\n
            **Loan_Amount_Term** - Term of Loan in Months\n
            **Credit_History** - Credit History Meets Guidelines\n
            **Property_Area** - Urban/Semi Urban/Rural\n
            **Loan_Status** - Target variable, Loan Approved (Y/N)
            """)

    elif dataset == "Water Potability":

        st.warning("You selected **Water Potability** dataset")

        st.info("""
            **ph** - The ph Level of the water\n
            **Hardness** - Water hardness, a measure of mineral content\n
            **Solids** - Total dissolved solids in the water\n
            **Chloramines** - Chloramines concentration in the water\n
            **Sulfate** - Sulfate concentration in the water\n
            **Conductivity** - Electrical conductivity of the water\n
            **Organic_carbon** - Organic carbon content in the water\n
            **Trihalomethanes** - Trihalomethanes concentration in the water\n
            **Turbidity** - Turbidity level, a measure of water clarity\n
            **Potability** - Target variable, indicates water potability with values 1 (potable) and 0 (not potable)
            """)

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    dataset = st.selectbox("Select dataset",["Loan Prediction","Water Potability"])

    if dataset == "Loan Prediction":
        df = pd.read_csv("loan_pred.csv")
    elif dataset == "Water Potability":
        df = pd.read_csv("water_potability.csv")
    
    description_panel(df)

elif menu == "Modeling":

    st.header("Modeling")

    if not os.path.exists("model.csv"):
        st.header("Please Run Preprocessing")
    else:
        df = pd.read_csv("model.csv")
        st.dataframe(df)

        p1,p2 = st.columns(2)

        p1.subheader("Scaling")
        scaling_method = p1.radio("",["Standard","Robust","MinMax"])

        p2.subheader("Encoder")
        encoder_method = p2.radio("",["Label","One-Hot"])

        st.header("Train and Test Splitting")
        p1,p2 = st.columns(2)
        random_state = p1.text_input("Random State")
        test_size = p2.text_input("Test Size")

        model = st.selectbox("Select Model",["Xgboost","Catboost"])
        st.markdown("You selected **{0}** Model".format(model))

        if st.button("Run Model"):
            cat_cols = df.iloc[:,:-1].select_dtypes(include="object").columns
            num_cols = df.iloc[:,:-1].select_dtypes(exclude="object").columns

            y = df.iloc[:,[-1]]

            if num_cols.size>0:

                if scaling_method == "Standard":
                    sc = StandardScaler()
                elif scaling_method == "Robust":
                    sc = RobustScaler()
                elif scaling_method == "MinMax":
                    sc = MinMaxScaler()

                df[num_cols] = sc.fit_transform(df[num_cols])

            if cat_cols.size>0:

                if encoder_method == "Label":
                    lb = LabelEncoder()
                    for col in cat_cols:
                        df[col] = lb.fit_transform(df[col])

                elif encoder_method == "One-Hot":
                    df.drop(df.iloc[:,[-1]],axis=1,inplace=True)
                    dummy_df = df[cat_cols]
                    dummy_df = pd.get_dummies(dummy_df,drop_first=True)
                    df.drop(cat_cols,axis=1,inplace=True)
                    df = pd.concat([df,dummy_df,y],axis=1)

            st.dataframe(df)

            X = df.iloc[:,:-1]

            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=float(test_size),random_state=int(random_state),stratify=y)

            st.markdown("X_train size {0}".format(X_train.shape))
            st.markdown("X_test size {0}".format(X_test.shape))
            st.markdown("y_train size {0}".format(y_train.shape))
            st.markdown("y_test size {0}".format(y_test.shape))

            if model == "Xgboost":
                model = XGBClassifier().fit(X_train,y_train)
            elif model == "Catboost":
                model = CatBoostClassifier().fit(X_train,y_train)

            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:,1]

            st.markdown("Confusion Matrix")
            st.write(metrics.confusion_matrix(y_test,y_pred))

            cl_report = metrics.classification_report(y_test,y_pred,output_dict=True)
            df_report = pd.DataFrame(cl_report).transpose()

            st.dataframe(df_report)

            accuracy = str(round(metrics.accuracy_score(y_test,y_pred),2))

            st.markdown("Accuracy score: {0}".format(accuracy))