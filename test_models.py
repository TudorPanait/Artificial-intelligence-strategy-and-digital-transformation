import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import time

# Set page configuration
st.set_page_config(page_title="Job Change Prediction", layout="wide")

# Title and Description
st.title("Job Change Prediction: Machine Learning Project")
st.markdown("""
This application is designed to explore, test, and compare various machine learning algorithms for predicting whether a person is willing to change jobs. Each model is explained in detail, including how it works, why it was chosen, and its performance.
""")

# Section 1: Upload Datasets
st.header("1. Upload Datasets")
st.markdown("""Upload the training and testing datasets in CSV format. Ensure they match the structure described in the project requirements.""")

train_file = st.file_uploader("Upload Training Dataset", type="csv", key="train")
test_file = st.file_uploader("Upload Testing Dataset", type="csv", key="test")

if train_file and test_file:
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    st.success("Datasets uploaded successfully!")
    st.write("Training Dataset:", train_data.head())
    st.write("Testing Dataset:", test_data.head())

    # Preprocessing specific columns
    # Handle 'years_since_job_change'
    train_data['years_since_job_change'] = train_data['years_since_job_change'].replace({'>4': 5, 'never_changed': -1, 'unknown': -1}).astype(float)
    test_data['years_since_job_change'] = test_data['years_since_job_change'].replace({'>4': 5, 'never_changed': -1, 'unknown': -1}).astype(float)

    # Handle missing or unknown categorical data
    most_frequent_gender = train_data['gender'].mode()[0]
    train_data['gender'] = train_data['gender'].replace({'Unknown': most_frequent_gender})
    test_data['gender'] = test_data['gender'].replace({'Unknown': most_frequent_gender})

    # Encode categorical columns
    categorical_cols = ['gender', 'education', 'field_of_studies', 'is_studying', 'county', 'size_of_company', 'type_of_company']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_train = pd.DataFrame(encoder.fit_transform(train_data[categorical_cols]))
    encoded_test = pd.DataFrame(encoder.transform(test_data[categorical_cols]))

    # Fix column names after encoding
    encoded_train.columns = encoder.get_feature_names_out(categorical_cols)
    encoded_test.columns = encoder.get_feature_names_out(categorical_cols)

    # Merge with numerical columns
    num_cols = ['age', 'relative_wage', 'years_since_job_change', 'years_of_experience', 'hours_of_training', 'is_certified']
    X_train = pd.concat([train_data[num_cols], encoded_train], axis=1)
    X_test = pd.concat([test_data[num_cols], encoded_test], axis=1)

    # Ensure all numeric columns are properly handled
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    
    # Replace NaN with -1 (or use imputation as needed)
    X_train.fillna(-1, inplace=True)
    X_test.fillna(-1, inplace=True)

    y_train = train_data['willing_to_change_job'].map({'Yes': 1, 'No': 0})
else:
    st.warning("Please upload both the training and testing datasets to proceed.")

# Section 2: Model Selection and Parameters
if train_file and test_file:
    st.header("2. Model Selection and Parameters")
    st.markdown("Select the models you want to run and configure their parameters dynamically.")

    # Available models and default parameters
    model_configs = {
        "Logistic Regression": {
            "enabled": st.checkbox("Logistic Regression", value=True),
            "parameters": {
                "max_iter": st.number_input("Max Iterations (Logistic Regression)", min_value=100, max_value=10000, value=1000, step=100)
            }
        },
        "K-Nearest Neighbors": {
            "enabled": st.checkbox("K-Nearest Neighbors", value=False),
            "parameters": {
                "n_neighbors": st.number_input("Number of Neighbors (KNN)", min_value=1, max_value=50, value=5, step=1)
            }
        },
        "LASSO (L1 Regularization)": {
            "enabled": st.checkbox("LASSO (L1 Regularization)", value=False),
            "parameters": {
                "alpha": st.number_input("Alpha (LASSO)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            }
        },
        "Ridge (L2 Regularization)": {
            "enabled": st.checkbox("Ridge (L2 Regularization)", value=False),
            "parameters": {
                "alpha": st.number_input("Alpha (Ridge)", min_value=0.01, max_value=10.0, value=1.0, step=0.1)
            }
        },
        "Elastic Net": {
            "enabled": st.checkbox("Elastic Net", value=False),
            "parameters": {
                "alpha": st.number_input("Alpha (Elastic Net)", min_value=0.01, max_value=10.0, value=1.0, step=0.1),
                "l1_ratio": st.slider("L1 Ratio (Elastic Net)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            }
        },
        "Support Vector Machine (SVM)": {
            "enabled": st.checkbox("Support Vector Machine (SVM)", value=False),
            "parameters": {
                "C": st.number_input("C (SVM)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            }
        },
        "Decision Tree": {
            "enabled": st.checkbox("Decision Tree", value=False),
            "parameters": {
                "max_depth": st.number_input("Max Depth (Decision Tree)", min_value=1, max_value=50, value=10, step=1)
            }
        },
        "Random Forest": {
            "enabled": st.checkbox("Random Forest", value=False),
            "parameters": {
                "n_estimators": st.number_input("Number of Trees (Random Forest)", min_value=10, max_value=1000, value=100, step=10)
            }
        },
        "Gradient Boosting": {
            "enabled": st.checkbox("Gradient Boosting", value=False),
            "parameters": {
                "learning_rate": st.slider("Learning Rate (Gradient Boosting)", min_value=0.01, max_value=1.0, value=0.1, step=0.01),
                "n_estimators": st.number_input("Number of Trees (Gradient Boosting)", min_value=10, max_value=1000, value=100, step=10)
            }
        },
        "XGBoost": {
            "enabled": st.checkbox("XGBoost", value=False),
            "parameters": {
                "learning_rate": st.slider("Learning Rate (XGBoost)", min_value=0.01, max_value=1.0, value=0.1, step=0.01),
                "n_estimators": st.number_input("Number of Trees (XGBoost)", min_value=10, max_value=1000, value=100, step=10)
            }
        },
        "AdaBoost": {
            "enabled": st.checkbox("AdaBoost", value=False),
            "parameters": {
                "n_estimators": st.number_input("Number of Trees (AdaBoost)", min_value=10, max_value=1000, value=50, step=10)
            }
        },
        "CatBoost": {
            "enabled": st.checkbox("CatBoost", value=False),
            "parameters": {
                "learning_rate": st.slider("Learning Rate (CatBoost)", min_value=0.01, max_value=1.0, value=0.1, step=0.01),
                "iterations": st.number_input("Iterations (CatBoost)", min_value=10, max_value=1000, value=100, step=10)
            }
        },
        "Neural Network (MLP)": {
            "enabled": st.checkbox("Neural Network (MLP)", value=False),
            "parameters": {
                "hidden_layer_sizes": st.text_input("Hidden Layer Sizes (MLP, e.g., 100,100)", value="100,100"),
                "learning_rate_init": st.slider("Learning Rate (MLP)", min_value=0.001, max_value=0.1, value=0.001, step=0.001)
            }
        }
    }

    # Section 3: Stacking Configuration
    st.header("3. Stacking Configuration")
    st.markdown("Choose models to include in the stacking classifier and configure the final estimator.")

    stacking_models = [name for name, config in model_configs.items() if config["enabled"] and st.checkbox(f"Include {name} in Stacking", value=False)]
    final_estimator = st.selectbox("Final Estimator for Stacking", list(model_configs.keys()))

    # Section 4: Run Models
    st.header("4. Run Models")
    if st.button("Run Selected Models"):
        performance = []
        with st.spinner("Training selected models..."):
            for name, config in model_configs.items():
                if config["enabled"]:
                    st.subheader(f"Training {name}")
                    st.write(f"Using parameters: {config['parameters']}")
                    # Dynamically create model with parameters
                    if name == "Logistic Regression":
                        model = LogisticRegression(max_iter=config['parameters']['max_iter'])
                    elif name == "K-Nearest Neighbors":
                        model = KNeighborsClassifier(n_neighbors=config['parameters']['n_neighbors'])
                    elif name == "LASSO (L1 Regularization)":
                        model = LogisticRegression(penalty='l1', solver='liblinear', C=config['parameters']['alpha'], max_iter=1000)
                    elif name == "Ridge (L2 Regularization)":
                        model = RidgeClassifier(alpha=config['parameters']['alpha'])
                    elif name == "Elastic Net":
                        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=config['parameters']['l1_ratio'], C=config['parameters']['alpha'], max_iter=1000)
                    elif name == "Support Vector Machine (SVM)":
                        model = SVC(C=config['parameters']['C'], probability=True)
                    elif name == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=config['parameters']['max_depth'])
                    elif name == "Random Forest":
                        model = RandomForestClassifier(n_estimators=config['parameters']['n_estimators'])
                    elif name == "Gradient Boosting":
                        model = GradientBoostingClassifier(learning_rate=config['parameters']['learning_rate'], n_estimators=config['parameters']['n_estimators'])
                    elif name == "XGBoost":
                        model = XGBClassifier(learning_rate=config['parameters']['learning_rate'], n_estimators=config['parameters']['n_estimators'], use_label_encoder=False, eval_metric='logloss')
                    elif name == "AdaBoost":
                        model = AdaBoostClassifier(n_estimators=config['parameters']['n_estimators'])
                    elif name == "CatBoost":
                        model = CatBoostClassifier(learning_rate=config['parameters']['learning_rate'], iterations=config['parameters']['iterations'], verbose=0)
                    elif name == "Neural Network (MLP)":
                        hidden_layers = tuple(map(int, config['parameters']['hidden_layer_sizes'].split(',')))
                        model = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=config['parameters']['learning_rate_init'], max_iter=1000)

                    # Train and evaluate
                    start_time = time.time()
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='balanced_accuracy')
                    elapsed_time = time.time() - start_time
                    mean_score = np.mean(scores)

                    performance.append({
                        "Model": name,
                        "Balanced Accuracy": mean_score,
                        "Time (s)": elapsed_time
                    })

                    st.write(f"Balanced Accuracy: {mean_score:.4f}")
                    st.write(f"Training Time: {elapsed_time:.2f} seconds")
                    
                    model.fit(X_train, y_train)
                    
                    # Generate predictions on the test set
                    test_data["willing_to_change_job"] = model.predict(X_test)

                    # Save predictions to a CSV file
                    file_name = f"test_predictions_{name}_{int(time.time())}.csv"
                    st.download_button(
                        label="Download Predictions",
                        data=test_data[["id", "willing_to_change_job"]].to_csv(index=False),
                        file_name=file_name,
                        mime="text/csv",
                        key=f"download_predictions_{name}_{int(time.time())}"
                    )

            # Stacking Model
            if stacking_models:
                st.subheader("Stacking Model")
                estimators = [(model_name, LogisticRegression()) for model_name in stacking_models]
                stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

                start_time = time.time()
                stacking_scores = cross_val_score(stacking_classifier, X_train, y_train, cv=5, scoring='balanced_accuracy')
                stacking_elapsed_time = time.time() - start_time
                stacking_mean_score = np.mean(stacking_scores)

                performance.append({
                    "Model": "Stacking Classifier",
                    "Balanced Accuracy": stacking_mean_score,
                    "Time (s)": stacking_elapsed_time
                })

                st.write(f"Balanced Accuracy (Stacking): {stacking_mean_score:.4f}")
                st.write(f"Training Time (Stacking): {stacking_elapsed_time:.2f} seconds")

        # Display Final Comparison
        st.subheader("Final Model Comparison")
        performance_df = pd.DataFrame(performance)
        st.write(performance_df)
        st.balloons()
        

