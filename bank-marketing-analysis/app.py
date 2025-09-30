import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ====== Page Configuration ======
st.set_page_config(
    page_title="Bank Marketing Campaign Analysis",
    page_icon="🏦",
    layout="wide"
)

# ====== Title ======
st.title("🏦 Bank Marketing Campaign Analysis")

# ====== Load Model ======
@st.cache_resource
def load_model():
    model_path = r"C:\Users\moham\Documents\Projects\python\ipynb\Analysis of Banking Marketing Campaigns\Analysis of Banking Marketing Campaigns.sav"
    try:
        model = pickle.load(open(model_path, "rb"))
        st.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")
        return None

model = load_model()

# Show model info if loaded
if model is not None:
    st.sidebar.info(f"📊 The model expects {model.n_features_in_} inputs")

# ====== Prediction Section ======
if model is not None:
    st.header("🔮 Prediction")
    st.info("""
    ⚠️ The model requires 16 inputs. 
    Please fill all fields below:
    """)
    
    # First row of inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        job = st.selectbox("Job", [
            "admin.", "technician", "services", "management", 
            "retired", "blue-collar", "entrepreneur", "housemaid",
            "unemployed", "self-employed", "student", "unknown"
        ])
    
    with col2:
        marital = st.selectbox("Marital Status", [
            "married", "single", "divorced"
        ])
        education = st.selectbox("Education Level", [
            "primary", "secondary", "tertiary", "unknown"
        ])
    
    with col3:
        default = st.selectbox("Default on Loan", ["no", "yes"])
        balance = st.number_input("Balance", value=1000, step=100)
    
    with col4:
        housing = st.selectbox("Housing Loan", ["yes", "no"])
        loan = st.selectbox("Personal Loan", ["yes", "no"])

    # Second row of inputs
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        contact = st.selectbox("Contact Type", [
            "cellular", "telephone", "unknown"
        ])
        day = st.number_input("Day of the Month", min_value=1, max_value=31, value=15)
    
    with col6:
        month = st.selectbox("Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        duration = st.number_input("Call Duration (seconds)", value=300, step=10)
    
    with col7:
        campaign = st.number_input("Number of Contacts", min_value=1, max_value=50, value=1)
        pdays = st.number_input("Days Since Previous Contact", value=-1)
    
    with col8:
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0)
        poutcome = st.selectbox("Previous Outcome", [
            "success", "failure", "other", "unknown"
        ])

    # ====== Prepare Input Data ======
    def prepare_input(age, job, marital, education, default, balance, 
                     housing, loan, contact, day, month, duration, 
                     campaign, pdays, previous, poutcome):
        """
        Encode user inputs into the format expected by the model
        """
        job_encoded = {
            "admin.": 0, "technician": 1, "services": 2, "management": 3,
            "retired": 4, "blue-collar": 5, "entrepreneur": 6, "housemaid": 7,
            "unemployed": 8, "self-employed": 9, "student": 10, "unknown": 11
        }.get(job, 11)
        
        marital_encoded = {"married": 0, "single": 1, "divorced": 2}.get(marital, 0)
        education_encoded = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 3}.get(education, 3)
        default_encoded = {"no": 0, "yes": 1}.get(default, 0)
        housing_encoded = {"no": 0, "yes": 1}.get(housing, 0)
        loan_encoded = {"no": 0, "yes": 1}.get(loan, 0)
        
        contact_encoded = {"cellular": 0, "telephone": 1, "unknown": 2}.get(contact, 2)
        
        month_encoded = {
            "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
            "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
        }.get(month, 0)
        
        poutcome_encoded = {"unknown": 0, "failure": 1, "other": 2, "success": 3}.get(poutcome, 0)

        input_data = [
            age, balance, day, duration, campaign, pdays, previous,
            job_encoded, marital_encoded, education_encoded, default_encoded,
            housing_encoded, loan_encoded, contact_encoded, month_encoded, poutcome_encoded
        ]
        
        return np.array([input_data])

    # ====== Run Prediction ======
    if st.button("🔄 Run Prediction", type="primary"):
        try:
            prediction_input = prepare_input(
                age, job, marital, education, default, balance,
                housing, loan, contact, day, month, duration,
                campaign, pdays, previous, poutcome
            )
            
            st.info(f"📤 Number of inputs sent: {len(prediction_input[0])}")
            
            prediction = model.predict(prediction_input)
            prediction_proba = model.predict_proba(prediction_input)
            
            st.success("🎯 Prediction Results:")
            
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                result = "Yes ✅" if prediction[0] == 1 else "No ❌"
                st.metric("Will Subscribe to Deposit?", result)
            
            with col_result2:
                confidence = max(prediction_proba[0])
                st.metric("Confidence Level", f"{confidence:.1%}")
            
            with col_result3:
                prob_yes = prediction_proba[0][1]
                prob_no = prediction_proba[0][0]
                st.metric("Probability of Acceptance", f"{prob_yes:.1%}")

            if prediction[0] == 1:
                st.balloons()
                st.success("🎉 This customer is a great candidate for the marketing campaign!")
                st.progress(float(prob_yes))
            else:
                st.warning("📉 This customer might need a different strategy")
                st.progress(float(prob_no))
                
            with st.expander("📊 Probability Details"):
                prob_df = pd.DataFrame({
                    'Result': ['Will Not Subscribe', 'Will Subscribe'],
                    'Probability': [prob_no, prob_yes]
                })
                st.bar_chart(prob_df.set_index('Result'))
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("""
            💡 **Suggested Fixes:**
            - Make sure data encoding matches the training process
            - Check the order of inputs
            - You may need to adjust `prepare_input`
            """)

# ====== Sidebar Help ======
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    **To avoid input mismatch issues:**
    
    1. **Correct number**: 16 inputs  
    2. **Correct order**: Must match training data  
    3. **Correct encoding**: Convert text to numeric values  
    
    **Required Inputs:**
    - Age, Balance, Day, Duration  
    - Campaign, Pdays, Previous contacts  
    - Job, Marital status, Education  
    - Default, Housing loan, Personal loan  
    - Contact type, Month, Previous outcome  
    """)

# ====== Model Exploration Section ======
if model is not None:
    with st.expander("🔍 Explore Model"):
        st.subheader("Model Information")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of expected inputs:** {model.n_features_in_}")
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importances")
            fi = pd.DataFrame({
                "Feature": range(len(model.feature_importances_)),
                "Importance": model.feature_importances_
            })
            st.bar_chart(fi.set_index("Feature"))

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Developed with Streamlit | Bank Marketing Data Analysis</p>
</div>
""", unsafe_allow_html=True)
