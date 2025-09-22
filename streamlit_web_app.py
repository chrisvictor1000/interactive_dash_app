import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.datasets import load_iris, load_diabetes, load_wine
import plotly.express as px
import plotly.graph_objects as go





#set the page configuration 
st.set_page_config(
    page_title="Data Science Process Demo",
    layout="wide"
)

#create title and description 
st.title("Data science process demonstration")
st.markdown(
    """ This app demonstrates the complete data science process from data loading to model deployment. """
)

### Side bar for Navigation 
st.sidebar.title("Navigation")
section  = st.sidebar.radio(
    "Select Section:",[
        "1. Data Loading",
        "2. Data Exploration",
        "3. Data Preprocessing",
        "4. Model Training",
        "5. Model Evaluation",
        "6. Prediction"
    ]
)

#load Sample dataset 
@st.cache_data
def load_sample_data():
    iris = load_iris()
    diabetes = load_diabetes()
    wine = load_wine()


    #create a dictionary of all datasets
    datasets = {
        "Iris Classification": pd.DataFrame(iris.data, columns=iris.feature_names),
        "Diabetes Regression": pd.DataFrame(diabetes.data, columns=diabetes.feature_names),
        "wine classification": pd.DataFrame(wine.data, columns=wine.feature_names)
    }

    #collect dictionary of target featurs 
    targets = {
        "Iris Classification": iris.target,
        "Diabetes Regression": diabetes.target,
        "wine classification": wine.target
    }

    return datasets, targets


datasets, targets = load_sample_data()

#section 1: Data Loading 
if section == "1. Data Loading":
    st.header("Data Loading")

    #lets create columns 
    col1, col2 = st.columns([2, 1])

    # we create a dataset selector to choose dataset from

    with col1:
        dataset_choice = st.selectbox(
            "choose a sample dataset:",
            list(datasets.keys())
        )

        #display the dataset info 
        df = datasets[dataset_choice]
        df["target"] = targets[dataset_choice]

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        #show the dataset statistics 
        st.subheader("Dataset Information:")
        st.write(f"shape: {df.shape}")
        st.write(f"Number of Features: {len(df.columns)}")
        st.write(f"Number of samples: {len(df)}")
    
    with col2:
        st.subheader("Dataset Information")
        st.info(f"Selected : {dataset_choice}")
        st.metric("Features : ", len(df.columns) - 1)
        st.metric("Samples :", len(df))
        st.metric("Missing Values", df.isnull().sum().sum())


        #show target distribution 
        if dataset_choice != "Diabetes Regression":
            target_counts = df["target"].value_counts()
            fig_pie = px.pie(values=target_counts.values, names=target_counts.index, title="Target Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)



#section 2: Data Exploration 
elif section == "2. Data Exploration":
    st.header("Data Exploration")
    dataset_choice = st.selectbox(
        'choose data exploration:',
        list(datasets.keys()),
        key="explore_dataset"
    )
    #create  a select box for chosing a dataset choice

    df = datasets[dataset_choice]
    df["target"] = targets[dataset_choice]

    #create tabs for basics
    tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Descriptive Stats", "Visualisation", "Correlation"])

    with tab1:
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        #data types 
        with col1:
            st.write("Data Type: **")
            st.write(df.dtypes)
        
        #missing values 
        
        with col2:
            st.write("Missing Values: **")
            missing_df = pd.DataFrame({
                "Missing Values": df.isnull().sum(),
                "percentage": (df.isnull().sum() / len(df)) * 100
            })
            st.dataframe(missing_df)   


    with tab2:
        st.subheader("Descriptive statistics")
        st.dataframe(df.describe())
        
    with tab3:
        st.subheader("Data Visualisation")
        col1, col2 = st.columns(2)

        with col1:
            # provide a chart for histogram

            feature_to_plot = st.selectbox("select feature for histogram:", df.columns[:-1])
            fig_hist = px.histogram(df, x=feature_to_plot, color="target", title=f"Distribution of  {feature_to_plot}")
            st.plotly_chart(fig_hist, use_container_width=True)  

        with col2:
            #plot a boxplot 
            fig_box = px.box(df, y=feature_to_plot, color="target", title=f"Boxplot of {feature_to_plot}")

    with tab4:
        st.subheader("Correlation Analysis")

        #calculate the correlation matrix
        corr_matrix = df.corr()

        fig_corr = px.imshow(corr_matrix,
                             text_auto=True,
                             aspect="auto",
                             title="correlation Matrix")
        st.plotly_chart(fig_corr, use_container_width=True)

    

#data preprocessing 
elif section == "3. Data Preprocessing":
    st.subheader("Data Preprocessing")
    dataset_choice = st.selectbox(
        "choose dataset",
        list(datasets.keys()),
        key="preprocess_data"
    )

    #create a selectbox for the datasets
    df = datasets[dataset_choice]
    target_values = targets[dataset_choice]

    st.subheader("Preprocessing Options")
    
    #create display columns 
    col1, col2, col3= st.columns(3)

    with col1:
        st.write("Feature Scaling")
        scale_features = st.checkbox("Apply Standard Scaling", value=True)

    with col2:
        st.write("Train-test-split")
        test_size = st.slider("test size", 0.1, 0.5, 0.2, 0.05)

    with col3:
        st.write("Random State")
        random_state = st.number_input("Random State", 0, 100, 42)

    
    #apply preprocessing 
    X = df.copy()
    y = target_values


    if scale_features:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        st.success("Features scaled using Standard Scaler")

        #perfrom a train-test split 
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

        st.subheader("preprocessing Results")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Training Set")
            st.write(f"Shape: {X_train.shape}")
            st.dataframe(X_train.head())

        with col2:
            st.write("Test Set")
            st.write(f"Shape: {X_test.shape}")
            st.dataframe(X_test.head())


#model Training 
elif section == "4. Model Training":
    st.header("Model Training")

    dataset_choice = st.selectbox(
        "Choose Dataset:",
        list(datasets.keys()),
        key="train_dataset"
    )

    df = datasets[dataset_choice]
    target_values = targets[dataset_choice]


    #determine the problem type 

    problem_type = "classification" if dataset_choice != "Diabetes Regression" else "regression"

    st.subheader("Model Selection")

    col1, col2 = st.columns(2)

    with col1:
        if problem_type == "classification":
            model_choice = st.selectbox(
                "choose a model:",
                ["Random Forest", "Logistic Regression"]
            )
        else:
            model_choice = st.selectbox(
                "choose a model:",
                ["Random Forest", "Linear Regression"]
            )

    with col2:
        st.write("Hyper parameters")
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees", 10, 200, 100)
            max_depth = st.slider("Max Depth", 2, 20, 10)

    
    #preprocess data
    X = df.copy()
    y = target_values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Train model
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if problem_type == "classification":
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                else:
                    model = LogisticRegression(random_state=42)
            else:
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                else:
                    model = LinearRegression()
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Store in session state
            st.session_state.model = model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.scaler = scaler
            st.session_state.problem_type = problem_type
            
            st.success("Model trained successfully!")

#Model Evaluation 
elif section == "5. Model Evaluation":
    st.header("Model Evaluation")

    if "model" not in st.session_state:
        st.warning("Please train a model in the 'Model Training' Section ")
    else:
        model = st.session_state.model 
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        problem_type = st.session_state.problem_type

        st.subheader("Performance Metrics")

        if problem_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            st.metric("Accuracy", f"{accuracy:.4f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Report**")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

            with col2:
                #confusion Matrix 
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    aspect="auto",
                    title="confusion Matrix"
                    )
                st.plotly_chart(fig_cm, use_container_width=True)

        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Mean Squared error", f"{mse:.4f}")
            with col2:
                st.metric("R2 Score", f'{r2:.4f}')

            #regression plot 
            fig_reg = px.scatter(
                x=y_test,
                y=y_pred,
                labels={'x':'actual', "y": "Predicted"},
                title="Actual vs Predicted values"
            )
            fig_reg.add_trace(go.Scatter(
                x=[min(y_test), max(y_test)],
                y= [min(y_test), max(y_test)],
                mode="lines", name="perfect fit"
            ))

            st.plotly_chart(fig_reg, use_container_width=True)




#section 6: PREDICTIONS

# Section 6: Prediction
elif section == "6. Prediction":
    st.header("🔮 Make Predictions")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' section.")
    else:
        model = st.session_state.model
        scaler = st.session_state.scaler
        problem_type = st.session_state.problem_type
        
        st.subheader("Input Features for Prediction")
        
        # Create input fields for each feature
        feature_inputs = {}
        df = datasets[list(datasets.keys())[0]]  # Use first dataset for feature names
        
        cols = st.columns(3)
        for i, feature in enumerate(df.columns):
            with cols[i % 3]:
                feature_inputs[feature] = st.number_input(
                    f"{feature}",
                    value=float(df[feature].mean()),
                    step=0.1
                )
        
        if st.button("Predict"):
            # Prepare input data
            input_data = pd.DataFrame([feature_inputs])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            if problem_type == "classification":
                st.success(f"Predicted Class: {prediction}")
            else:
                st.success(f"Predicted Value: {prediction:.4f}")
            
            # Show prediction probabilities for classification
            if problem_type == "classification" and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_scaled)[0]
                prob_df = pd.DataFrame({
                    'Class': range(len(probabilities)),
                    'Probability': probabilities
                })
                fig_probs = px.bar(prob_df, x='Class', y='Probability', 
                                 title="Prediction Probabilities")
                st.plotly_chart(fig_probs, use_container_width=True)




# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Data Science Process:**
1. 📁 Data Loading
2. 🔍 Data Exploration  
3. ⚙️ Data Preprocessing
4. 🤖 Model Training
5. 📈 Model Evaluation
6. 🔮 Prediction
""")

