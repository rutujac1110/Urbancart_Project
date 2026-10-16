"""
UrbanKart Customer Feedback Analysis
Standalone Streamlit page. Keep this file in the project's root directory.
Run with: streamlit run Customer_Feedback_Analysis.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATASET_NAME = "HuggingFace_NLP_Customer Reviews.csv"
ROOT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ROOT_DIR / DATASET_NAME
REVIEW_COLUMN = "review_text"
BATCH_SIZE = 16

st.set_page_config(
    page_title="UrbanKart | Customer Feedback Analysis",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #245b78);
        color: white; margin-bottom: 1.2rem;
      }
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
      .hero p {margin: 0; color: #e5eef5;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.25);
        padding: 1rem; border-radius: 12px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>UrbanKart Customer Intelligence Hub</h1>
      <p>Customer Feedback Analysis · Transformer-based sentiment predictions</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Model: cardiffnlp/twitter-roberta-base-sentiment-latest. "
    "Predictions are estimates and may not always reflect the reviewer's intent."
)


@st.cache_resource(show_spinner="Loading the sentiment model…")
def load_sentiment_model():
    """Load the Hugging Face model once per Streamlit server session."""
    import torch
    import transformers
    import transformers.pipelines as transformers_pipelines
    from transformers import pipeline

    # Compatibility workaround for Transformers builds where the pipelines
    # module references torch internally without defining its module-level name.
    transformers_pipelines.torch = torch

    analyzer = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=-1,  # CPU: avoids requiring a CUDA setup
    )
    return analyzer


def normalize_sentiment(label):
    """Normalize model labels for consistent dashboard grouping."""
    label = str(label).strip().lower()
    if label in {"label_0", "negative"}:
        return "negative"
    if label in {"label_1", "neutral"}:
        return "neutral"
    if label in {"label_2", "positive"}:
        return "positive"
    return label


def find_review_column(dataframe):
    """Find review_text without depending on exact capitalization or spaces."""
    lookup = {str(column).strip().lower(): column for column in dataframe.columns}
    return lookup.get(REVIEW_COLUMN)


def analyze_texts(analyzer, texts):
    """Run predictions in batches and return sentiment labels and confidence."""
    predictions = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        batch_results = analyzer(
            batch,
            batch_size=BATCH_SIZE,
            truncation=True,
            max_length=512,
        )
        predictions.extend(batch_results)
    return predictions


def read_csv_safely(uploaded_file=None):
    """Read the uploaded CSV or the expected CSV in the root folder."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return None


try:
    sentiment_analyzer = load_sentiment_model()
except Exception as exc:
    st.error(
        "The sentiment model could not be loaded. Check that PyTorch and "
        "Transformers are installed in the same Python environment used to run "
        "Streamlit, and confirm that the first model download can access the internet."
    )
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

st.success("Sentiment model loaded and ready.")

tab_single, tab_batch = st.tabs(["✍️ Analyze a review", "📊 Analyze the CSV dataset"])

with tab_single:
    st.subheader("Analyze an individual customer review")
    review_text = st.text_area(
        "Customer review",
        placeholder="Paste a customer review here…",
        height=150,
        key="single_review_text",
    )
    if st.button("Analyze sentiment", type="primary", key="analyze_single"):
        if not review_text.strip():
            st.warning("Please enter a review before analyzing.")
        else:
            try:
                with st.spinner("Analyzing review…"):
                    result = sentiment_analyzer(
                        review_text.strip(),
                        truncation=True,
                        max_length=512,
                    )[0]
                label = normalize_sentiment(result["label"])
                confidence = float(result["score"])
                st.divider()
                col1, col2 = st.columns(2)
                col1.metric("Predicted sentiment", label.title())
                col2.metric("Confidence score", f"{confidence:.1%}")
                st.progress(max(0.0, min(confidence, 1.0)))
                st.caption(
                    "Confidence is the model's score for its selected label, "
                    "not a guarantee that the prediction is correct."
                )
            except Exception as exc:
                st.error("This review could not be analyzed.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

with tab_batch:
    st.subheader("Batch sentiment analysis")
    st.write(
        "The app looks for the dataset in the same folder as this Python file. "
        "If it is stored elsewhere, upload it below."
    )
    uploaded_csv = st.file_uploader(
        "Upload HuggingFace_NLP_Customer Reviews.csv",
        type=["csv"],
        help="The CSV must include a review_text column.",
        key="feedback_csv_upload",
    )

    try:
        source_df = read_csv_safely(uploaded_csv)
    except Exception as exc:
        source_df = None
        st.error("The CSV could not be read. Check that it is a valid CSV file.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

    if source_df is None:
        st.info(
            f"No dataset found at `{DATASET_PATH.name}` in the project root. "
            "Place the CSV beside this script or upload it above."
        )
    else:
        review_col = find_review_column(source_df)
        st.caption(
            f"Loaded {len(source_df):,} rows and {len(source_df.columns):,} columns."
        )
        st.dataframe(source_df.head(5), use_container_width=True)

        if review_col is None:
            st.error(
                "Could not find the required `review_text` column. "
                f"Columns found: {', '.join(map(str, source_df.columns))}"
            )
        else:
            valid_mask = source_df[review_col].notna() & (
                source_df[review_col].astype(str).str.strip() != ""
            )
            valid_count = int(valid_mask.sum())
            st.metric("Non-empty reviews available", f"{valid_count:,}")

            if valid_count == 0:
                st.warning("The dataset does not contain any non-empty reviews.")
            elif st.button(
                "Run batch sentiment analysis",
                type="primary",
                key="run_batch_analysis",
            ):
                result_df = source_df.copy()
                result_df["predicted_sentiment"] = pd.NA
                result_df["sentiment_confidence"] = pd.NA

                review_indices = result_df.index[valid_mask].tolist()
                review_texts = (
                    result_df.loc[valid_mask, review_col].astype(str).tolist()
                )
                progress = st.progress(0, text="Preparing reviews…")
                status = st.empty()

                try:
                    predictions = []
                    total = len(review_texts)
                    for start in range(0, total, BATCH_SIZE):
                        batch = review_texts[start : start + BATCH_SIZE]
                        predictions.extend(
                            sentiment_analyzer(
                                batch,
                                batch_size=BATCH_SIZE,
                                truncation=True,
                                max_length=512,
                            )
                        )
                        completed = min(start + len(batch), total)
                        progress.progress(
                            completed / total,
                            text=f"Analyzed {completed:,} of {total:,} reviews",
                        )
                        status.write(f"Processing review {completed:,} of {total:,}…")

                    result_df.loc[review_indices, "predicted_sentiment"] = [
                        normalize_sentiment(item["label"]) for item in predictions
                    ]
                    result_df.loc[review_indices, "sentiment_confidence"] = [
                        round(float(item["score"]), 6) for item in predictions
                    ]

                    st.session_state["urbankart_feedback_results"] = result_df
                    status.empty()
                    progress.empty()
                    st.success(f"Analysis completed for {total:,} reviews.")
                except Exception as exc:
                    status.empty()
                    progress.empty()
                    st.error(
                        "Batch analysis stopped before completion. "
                        "Try a smaller CSV or check the model/runtime error."
                    )
                    with st.expander("Technical details"):
                        st.code(f"{type(exc).__name__}: {exc}")

    results_df = st.session_state.get("urbankart_feedback_results")
    if results_df is not None:
        st.divider()
        st.subheader("Sentiment distribution")

        analyzed = results_df["predicted_sentiment"].dropna()
        if analyzed.empty:
            st.info("No sentiment predictions are available yet.")
        else:
            counts = (
                analyzed.value_counts()
                .reindex(["positive", "neutral", "negative"], fill_value=0)
                .rename_axis("Sentiment")
                .reset_index(name="Reviews")
            )
            total_analyzed = int(counts["Reviews"].sum())
            counts["Percentage"] = (
                counts["Reviews"] / total_analyzed * 100
            ).round(2)

            metric_cols = st.columns(3)
            for column, sentiment in zip(
                metric_cols, ["positive", "neutral", "negative"]
            ):
                count = int(
                    counts.loc[counts["Sentiment"] == sentiment, "Reviews"].iloc[0]
                )
                pct = count / total_analyzed * 100 if total_analyzed else 0
                column.metric(sentiment.title(), f"{count:,}", f"{pct:.1f}%")

            chart_data = counts.set_index("Sentiment")[["Reviews"]]
            st.bar_chart(chart_data, use_container_width=True)
            st.dataframe(counts, hide_index=True, use_container_width=True)

            st.subheader("Analyzed review records")
            st.dataframe(results_df.head(100), use_container_width=True)
            csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download analyzed CSV",
                data=csv_bytes,
                file_name="UrbanKart_Customer_Feedback_Analysis.csv",
                mime="text/csv",
                type="primary",
                use_container_width=False,
            )

st.divider()
st.caption(
    "Note: the model is pretrained on Twitter text. Customer-review predictions "
    "should be reviewed for domain fit before business decisions are made."
)
