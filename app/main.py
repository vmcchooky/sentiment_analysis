
import streamlit as st

# Đặt page config ngay đầu file
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide"
)

# Import các trang
from pages import sa_text, sa_file, our_final_project

def load_page(page):
    if page == "Home":
        show_home()
    elif page == "sa_text":
        sa_text.app()
    elif page == "sa_file":
        sa_file.app()
    elif page == "our_final_project":
        our_final_project.app()

def show_home():
    st.title("Welcome to Sentiment Analysis Dashboard 🎭")
    st.markdown("""
        ### Explore Sentiment Analysis Tools:
        - **sa_text**: Input a single text to analyze its sentiment (Positive or Negative).
        - **sa_file**: Upload a `.CSV` or `.XLSX` file with a 'text' column for batch sentiment analysis.
        - **our_final_project**: Presentation of our sentiment analysis project using BERT and Logistic Regression.
        """
    )
    st.image(
        "./img/pos_neg.png",  # Đường dẫn cập nhật dựa trên cấu trúc thư mục
        caption="Overview of Sentiment Analysis",
        use_container_width=True
    )
    st.markdown("---")
    st.markdown("Navigate through the **sidebar menu** on the left to choose a feature or service for sentiment analysis.")

def main():
    show_home()

if __name__ == "__main__":
    main()
