import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Streamlit app
st.title("Job Change Prediction App")

# File upload
st.subheader("Upload Training and Test Data")
training_file = st.file_uploader("Upload Training CSV", type=["csv"], key="training")
test_file = st.file_uploader("Upload Test CSV", type=["csv"], key="test")

if training_file and test_file:
    # Load data
    train_data = pd.read_csv(training_file)
    test_data = pd.read_csv(test_file)

    st.write("Training Data Preview:")
    st.write(train_data.head())

    st.write("Testing Data Preview:")
    st.write(test_data.head())

    # Define features and target
    features = [
        "gender", "age", "education", "field_of_studies", "is_studying",
        "county", "relative_wage", "years_since_job_change", "years_of_experience",
        "hours_of_training", "is_certified", "size_of_company", "type_of_company"
    ]
    target = "willing_to_change_job"

    # Encode categorical variables
    encoder = LabelEncoder()
    for col in train_data.columns:
        if train_data[col].dtype == 'object':
            train_data[col] = encoder.fit_transform(train_data[col].astype(str))
    for col in test_data.columns:
        if test_data[col].dtype == 'object':
            test_data[col] = encoder.fit_transform(test_data[col].astype(str))

    # Split training data
    X_train = train_data[features]
    y_train = train_data[target]

    # Prepare test data
    X_test = test_data[features]

    # Train Decision Tree model
    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on training data
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    st.write(f"Training Accuracy: {train_accuracy:.2f}")

    # Predict on test data
    test_data['Predicted'] = model.predict(X_test)

    # Save predictions to CSV
    output_file = "predictions.csv"
    predictions = test_data[["id", "Predicted"]]
    predictions.rename(columns={"Predicted": "willing_to_change_job"}, inplace=True)
    predictions.to_csv(output_file, index=False)

    st.write("Predictions saved to 'predictions.csv'.")

    # Provide download link
    with open(output_file, "rb") as file:
        st.download_button(
            label="Download Predictions CSV",
            data=file,
            file_name="predictions.csv",
            mime="text/csv"
        )
