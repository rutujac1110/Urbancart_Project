
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="UrbanKart Customer Intelligence Hub",
    page_icon="🛍️",
    layout="wide"
)

# --------------------------------------------------
# 2. LOAD SAVED MODEL
# --------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "urbankart_best_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    best_model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Save "
        "'urbankart_best_model.pkl' in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# --------------------------------------------------
# 3. GET FEATURE INFORMATION FROM THE PIPELINE
# --------------------------------------------------
try:
    preprocessor = best_model.named_steps["preprocessor"]

    num_pipeline = preprocessor.named_transformers_["num"]
    cat_pipeline = preprocessor.named_transformers_["cat"]

    num_cols = list(
        preprocessor.transformers_[0][2]
    )
    cat_cols = list(
        preprocessor.transformers_[1][2]
    )

    num_defaults = dict(zip(
        num_cols,
        num_pipeline.named_steps["imputer"].statistics_
    ))

    encoder = cat_pipeline.named_steps["encoder"]

    cat_options = {
        col: list(categories)
        for col, categories in zip(
            cat_cols, encoder.categories_
        )
    }

except Exception as e:
    st.error(
        "The saved model does not have the expected "
        "preprocessing pipeline. Please check the model "
        "training code."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# 4. APP HEADER
# --------------------------------------------------
st.title("🛍️ UrbanKart Customer Intelligence Hub")

st.markdown(
    """
    ### Customer Value Prediction

    Enter a customer's profile to estimate their
    **Customer Lifetime Value (CLV)** using machine learning.
    """
)

st.divider()

# --------------------------------------------------
# 5. CUSTOMER PROFILE INPUT FORM
# --------------------------------------------------
st.subheader("👤 Customer Profile")

input_values = {}

with st.form("customer_profile_form"):

    st.markdown("#### Customer Information")

    columns = st.columns(2)

    for i, col in enumerate(num_cols):
        with columns[i % 2]:
            default = num_defaults[col]

            if not np.isfinite(default):
                default = 0.0

            input_values[col] = st.number_input(
                label=col.replace("_", " "),
                value=float(default),
                format="%.2f",
                key=f"num_{col}"
            )

    for i, col in enumerate(cat_cols):
        with columns[i % 2]:
            options = cat_options[col]

            if options:
                input_values[col] = st.selectbox(
                    label=col.replace("_", " "),
                    options=options,
                    format_func=str,
                    key=f"cat_{col}"
                )
            else:
                input_values[col] = None

    submitted = st.form_submit_button(
        "🔮 Predict Customer Lifetime Value",
        use_container_width=True
    )

# --------------------------------------------------
# 6. MAKE PREDICTION
# --------------------------------------------------
if submitted:

    input_df = pd.DataFrame(
        [input_values],
        columns=num_cols + cat_cols
    )

    try:
        prediction = best_model.predict(input_df)[0]

        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric(
            label="Estimated Customer Lifetime Value",
            value=f"{prediction:,.2f}"
        )

        st.success(
            "Prediction generated successfully!"
        )

        st.caption(
            "Model prediction is an estimate, not a guarantee. "
            "Actual customer value may differ."
        )

    except Exception as e:
        st.error("Unable to generate the prediction.")
        st.exception(e)

# --------------------------------------------------
# 7. MODEL INFORMATION
# --------------------------------------------------
with st.expander("ℹ️ About this model"):

    st.write(
        "This application uses a trained regression model "
        "with preprocessing for numerical and categorical "
        "customer features."
    )

    st.write(
        "**Prediction target:** Customer Lifetime Value "
        "(`Customer_Lifetime_Value`)"
    )

    st.write(
        "**Features used:**",
        ", ".join(num_cols + cat_cols)
    )

    st.warning(
        "Predictions depend on the quality and range of "
        "the training data. They should not be treated "
        "as guaranteed customer outcomes."
    )
