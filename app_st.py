import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QWidget, QSlider, QLabel
from PySide6.QtCore import Qt
import faiss
import sys
import streamlit as st 

book_embeddings = np.load('book_embeddings.npy')

books = pd.read_csv('Books_rating.csv')

def glove_load(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Load GloVe 
glove_embedding = glove_load('glove.6B.100d.txt')

def get_sentence_embedding(sentence, embeddings_index, embedding_dim=100):
    words = sentence.split()
    valid_words = [word for word in words if word in embeddings_index]
    if not valid_words:
        return np.zeros(embedding_dim)
    embeddings = np.array([embeddings_index[word] for word in valid_words])
    return np.mean(embeddings, axis=0)

def get_book_recommendations(input_text, top_k=10):
    input_embedding = get_sentence_embedding(input_text, glove_embedding)
    index = faiss.IndexFlatL2(book_embeddings.shape[1])
    index.add(book_embeddings)
    _, top_indices = index.search(np.array([input_embedding]), top_k + 1)  # +1 to exclude the input itself

    recommendations = []
    seen_titles = set()
    for idx in top_indices[0][1:]: 
        book_info = books.iloc[idx]
        title = book_info['Title']
        if title not in seen_titles:
            seen_titles.add(title)
            recommendations.append({
                "Title": title,
                "Score": book_info['review/score'],
                "Price": book_info.get('Price', 'N/A'),
                "Similarity": np.dot(input_embedding, book_embeddings[idx]) / (np.linalg.norm(input_embedding) * np.linalg.norm(book_embeddings[idx]))
            })
        if len(recommendations) >= top_k:
            break
    
    return recommendations

st.title("Book Recommendation APP")
book_name_or_overview = st.text_input("Enter book name or overview")
if st.button("Get Recommendations"):
    if book_name_or_overview:
        recommendations = get_book_recommendations(book_name_or_overview)
        st.table(pd.DataFrame(recommendations))
    else:
        st.write("Please enter a book name or overview")


