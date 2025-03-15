import streamlit as st
import numpy as np
from joblib import load
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os

# Tải các tài nguyên NLTK cần thiết
nltk.download('punkt')

# Cache model loading
@st.cache_resource
def load_models():
    path = "C:/Users/vmcch/BIGDATA/sentiment_analysis_project/"  # Đường dẫn đã cập nhật
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
        # Nếu không có mô hình w2v đã huấn luyện trước, bạn có thể dùng mô hình pretrained từ gensim
        st.info("Using pretrained Word2Vec model (Google News).")
        import gensim.downloader as api
        w2v_model = api.load("word2vec-google-news-300")  # Mô hình w2v pretrained từ Google News (300 chiều)
        st.info("Pretrained Word2Vec model loaded successfully.")

    return lr_model, scaler, w2v_model

# Helper function để tạo w2v embeddings
def extract_w2v_embeddings(text, w2v_model, vector_size=300):
    try:
        # Tokenize văn bản
        tokens = word_tokenize(text.lower())
        st.info("Tokenization successful.")
    except Exception as e:
        st.error(f"Error during tokenization: {e}")
        raise

    # Tạo vector w2v bằng cách lấy trung bình vector của các từ
    vectors = []
    for token in tokens:
        try:
            if token in w2v_model:
                vectors.append(w2v_model[token])
        except KeyError:
            continue

    if not vectors:  # Nếu không có từ nào trong từ điển w2v
        st.warning("No valid words found in the input text for Word2Vec embedding.")
        return np.zeros(vector_size)

    try:
        embedding = np.mean(vectors, axis=0)
        st.info("Word2Vec embedding extracted successfully.")
    except Exception as e:
        st.error(f"Error extracting Word2Vec embedding: {e}")
        raise

    return embedding

def app():
    st.title("Phân tích cảm xúc - Nhập văn bản")

    try:
        lr_model, scaler, w2v_model = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return

    user_input = st.text_area("Nhập văn bản của bạn:", height=100)

    if st.button("Send"):
        if user_input:
            try:
                user_input_cleaned = user_input.strip().lower()
                st.info(f"Input standardized: {user_input_cleaned}")

                # Tạo w2v embedding cho văn bản đầu vào
                embedding = extract_w2v_embeddings(user_input_cleaned, w2v_model, vector_size=300)
                embedding = embedding.reshape(1, -1)  # Reshape thành (1, vector_size) để phù hợp với scaler

                # Chuẩn hóa embedding
                embedding_scaled = scaler.transform(embedding)

                # Dự đoán
                try:
                    sentiment_probs = lr_model.predict_proba(embedding_scaled)[0]
                    sentiment = lr_model.predict(embedding_scaled)[0]
                    st.info("Prediction successful.")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    return

                # Hiển thị kết quả
                # Giả sử nhãn 0 là tiêu cực và 1 là tích cực (dựa trên code cũ của bạn)
                st.success(f"Kết quả phân tích cảm xúc: {sentiment_probs[0] * 100:.2f}% tiêu cực, {sentiment_probs[1] * 100:.2f}% tích cực")
                st.write(f"Nhãn dự đoán: {'Tiêu cực' if sentiment == 0 else 'Tích cực'}")

            except Exception as e:
                st.error(f"Lỗi khi xử lý văn bản: {str(e)}")
        else:
            st.warning("Vui lòng nhập văn bản để phân tích!")

if __name__ == "__main__":
    app()