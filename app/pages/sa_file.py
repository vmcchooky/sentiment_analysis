import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Tải các tài nguyên NLTK cần thiết
nltk.download('punkt')

@st.cache_resource
def load_models():
    path = "/Users/ngoctrinh/Documents/SentimentAnalysis/"  # Đường dẫn đã cập nhật
    """Cache model loading to prevent Streamlit from reloading on each interaction"""
    try:
        lr_model = load(path + "model/logistic_regression_w2v_model.joblib")
        st.info("Logistic Regression model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Logistic Regression model: {e}")
        raise

    try:
        scaler = load(path + "model/scaler_w2v.joblib")
        st.info("Scaler loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Scaler: {e}")
        raise

    # Load mô hình Word2Vec đã được huấn luyện trước
    try:
        # Nếu bạn có mô hình w2v đã huấn luyện trước (tương tự mô hình tạo embeddings trong file .parquet)
        # Thay đường dẫn này bằng đường dẫn đến file mô hình w2v của bạn (nếu có)
        w2v_model = Word2Vec.load("path_to_your_w2v_model.model")
        st.info("Word2Vec model loaded successfully.")
    except Exception as e:
        # Nếu không có mô hình w2v đã huấn luyện trước, dùng mô hình pretrained từ gensim
        st.info("Using pretrained Word2Vec model (Google News).")
        import gensim.downloader as api
        w2v_model = api.load("word2vec-google-news-300")  # Mô hình w2v pretrained từ Google News (300 chiều)
        st.info("Pretrained Word2Vec model loaded successfully.")

    return lr_model, scaler, w2v_model

# Load models
lr_model, scaler, w2v_model = load_models()

def standardize_text(text):
    return str(text).strip().lower()

def extract_w2v_embeddings(text, w2v_model, vector_size=300):
    """Tạo w2v embedding cho văn bản bằng cách lấy trung bình vector của các từ"""
    try:
        # Tokenize văn bản
        tokens = word_tokenize(text.lower())
    except Exception as e:
        st.error(f"Error during tokenization: {e}")
        return np.zeros(vector_size)

    # Tạo vector w2v bằng cách lấy trung bình vector của các từ
    vectors = []
    for token in tokens:
        try:
            if token in w2v_model:
                vectors.append(w2v_model[token])
        except KeyError:
            continue

    if not vectors:  # Nếu không có từ nào trong từ điển w2v
        return np.zeros(vector_size)

    try:
        embedding = np.mean(vectors, axis=0)
    except Exception as e:
        st.error(f"Error extracting Word2Vec embedding: {e}")
        return np.zeros(vector_size)

    return embedding

def app():
    st.title("Sentiment Analysis - File Input")
    uploaded_file = st.file_uploader("Upload your file (.CSV or .XLSX)", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "text" not in df.columns:
            st.error("The file must have a 'text' column!")
            return

        if st.button("Analyze"):
            try:
                df["processed_text"] = df["text"].apply(standardize_text)
                df["embeddings"] = df["processed_text"].apply(lambda x: extract_w2v_embeddings(x, w2v_model, vector_size=300))

                # Chuẩn hóa embeddings
                df["embeddings"] = df["embeddings"].apply(lambda x: scaler.transform([x])[0])

                # Dự đoán cảm xúc và tính phần trăm
                df["sentiment_probs"] = df["embeddings"].apply(lambda x: lr_model.predict_proba([x])[0])
                df["negative_percentage"] = df["sentiment_probs"].apply(lambda x: x[0] * 100)
                df["positive_percentage"] = df["sentiment_probs"].apply(lambda x: x[1] * 100)

                # Hiển thị kết quả
                st.dataframe(df[["text", "negative_percentage", "positive_percentage"]])

                # Visualize sentiment distribution (dựa trên positive_percentage)
                sentiment_counts = df["positive_percentage"].value_counts(bins=10, sort=False)
                fig, ax = plt.subplots()
                ax.bar(sentiment_counts.index.astype(str), sentiment_counts.values)
                ax.set_xlabel("Positive Percentage Range")
                ax.set_ylabel("Count")
                ax.set_title("Sentiment Distribution")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during file analysis: {str(e)}")

if __name__ == "__main__":
    app()