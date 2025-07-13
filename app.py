import streamlit as st
import pandas as pd

# ✅ Set page config FIRST (Only ONCE!)
st.set_page_config(page_title="Cervical Cancer Prediction", layout="centered")

st.title("🧬 Cervical Cancer Prediction System")

uploaded_file = st.file_uploader("📂 Upload your cervical cancer dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, na_values='?')

    st.success("✅ File uploaded successfully!")
    st.write("📊 Dataset Preview:")
    st.write(df.head())
    
    st.write("🔍 Dataset Info:")
    st.text(df.info())

    # Import other modules only after the file is uploaded
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import joblib

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].mean(), inplace=True)

    # Label Encoding
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # Models
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # Train models and store accuracy
    model_accuracy = {}
    trained_models = {}

    st.subheader("🧠 Model Performance")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_accuracy[name] = acc
        trained_models[name] = model
        st.write(f"**{name}** Accuracy: {acc:.4f}")
        st.text(classification_report(y_test, y_pred))
        joblib.dump(model, f"{name.replace(' ', '_')}.pkl")

    # Sample Prediction Section
    st.subheader("🔍 Predict on Sample Data")

    selected_model = st.selectbox("Choose Model", list(trained_models.keys()))
    sample_data = np.zeros((1, X.shape[1]))
    sample_data[0, :] = X.mean().values
    sample_data_scaled = scaler.transform(sample_data)

    if st.button("Predict"):
        prediction = trained_models[selected_model].predict(sample_data_scaled)
        result = "🛑 High Risk Detected" if prediction[0] == 1 else "✅ Low Risk Detected"
        st.success(f"{selected_model} Prediction: {result}")

    # Health Disclaimer
    st.markdown("---")
    st.markdown(
        """
        ⚠️ **Disclaimer:**  
        This tool is for **educational/demo purposes only** and not a substitute for medical advice.  
        Always consult a licensed medical professional for any health concerns.
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
