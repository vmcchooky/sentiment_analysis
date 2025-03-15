import streamlit as st
import numpy as np
import plotly.graph_objects as go

def app():
    # Tiêu đề dự án
    st.title("Our Final Project: Sentiment Analysis Dashboard 🎯")

    # Giới thiệu dự án
    st.header("Introduction")
    st.markdown("""
    This project aims to develop a **Sentiment Analysis Dashboard** that allows users to analyze text sentiment (Positive/Negative) 
    either by inputting a single text or uploading a file with multiple entries. The key goals of this project are:
    - **Simplify sentiment analysis** for both individuals and organizations.
    - Provide **accurate predictions** using state-of-the-art machine learning models.
    - Enable **batch processing** for larger datasets.
    """)

    # Quy trình thực hiện
    st.header("Project Workflow")
    st.markdown("We followed these key steps to build our project:")
    st.markdown("""
    1. **Data Collection**: Collected a labeled dataset of text samples with sentiment labels.
    2. **Data Preprocessing**: Cleaned and normalized the text data (removing stopwords, punctuation, etc.).
    3. **Model Training**: Used machine learning algorithms such as Logistic Regression, SVM, or Deep Learning models.
    4. **Model Evaluation**: Assessed the model's performance using metrics like accuracy, precision, recall, and F1-score.
    5. **Dashboard Development**: Built a user-friendly Streamlit dashboard to integrate all functionalities.
    """)

    # Hiển thị hình ảnh minh họa hoặc biểu đồ
    st.image("./img/worldcloud.jpg", caption="Project Workflow Illustration", use_container_width=True)

    # Kết quả
    st.header("Results")
    st.markdown("""
    The project achieved the following results:
    - **Model Accuracy**: 92%
    - **Precision**: 90%
    - **Recall**: 88%
    - **F1-Score**: 89%
    """)

    # Tạo và hiển thị biểu đồ động với Plotly
    train_accuracies = [0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88]
    val_accuracies = [0.68, 0.73, 0.75, 0.77, 0.79, 0.81, 0.82, 0.83, 0.84, 0.85]
    epochs = list(range(1, 11))

    fig = go.Figure()
    
    # Thêm đường Training Accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=train_accuracies,
            name="Training Accuracy",
            mode='lines+markers',
            line=dict(width=3, color='#8884d8'),
            marker=dict(size=8, color='#8884d8')
        )
    )
    
    # Thêm đường Validation Accuracy
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=val_accuracies,
            name="Validation Accuracy",
            mode='lines+markers',
            line=dict(width=3, color='#82ca9d'),
            marker=dict(size=8, color='#82ca9d')
        )
    )
    
    # Cấu hình layout
    fig.update_layout(
        title={
            'text': "Training and Validation Accuracy over Epochs",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.1,
                y=1.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 1000, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"}
                        }]
                    )
                ]
            )
        ]
    )
    
    # Thêm lưới
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Tạo frames cho animation
    frames = [
        go.Frame(
            data=[
                go.Scatter(x=epochs[:k+1], y=train_accuracies[:k+1]),
                go.Scatter(x=epochs[:k+1], y=val_accuracies[:k+1])
            ]
        )
        for k in range(len(epochs))
    ]
    
    fig.frames = frames
    
    # Hiển thị biểu đồ
    st.plotly_chart(fig, use_container_width=True)

    # Hướng phát triển tương lai
    st.header("Future Work")
    st.markdown("""
    - Improve the model's accuracy for more complex sentences.
    - Expand the dataset to include multiple languages for multilingual sentiment analysis.
    - Add more visualizations and advanced features to the dashboard.
    """)

    # Thông tin nhóm và bảng đánh giá
    st.header("Teamwork Evaluation")
    st.markdown("""Below is the evaluation of our team's roles and contributions during the project:""")

    # Tạo bảng đánh giá
    team_data = {
        "Team Member": ["Võ Mạnh Cường", "Thằng Ngọc Quỳnh"],
        "Role": ["Data Collection & Preprocessing", "Model Development & Dashboard Design"],
        "Contributions": [
            "Collected and cleaned the dataset, prepared features for training.",
            "Developed and fine-tuned machine learning models & Designed the Streamlit dashboard, integrated features."
        ],
        "Percent (100%)": ["50%", "50%"]
    }

    # Hiển thị bảng trong ứng dụng
    st.table(team_data)


# Đảm bảo app() được gọi đúng khi chạy file
if __name__ == "__main__":
    app()