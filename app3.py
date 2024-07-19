import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QWidget, QSlider, QLabel
from PySide6.QtCore import Qt
import sys

def glove_load(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

# Load GloVe embeddings
glove_embedding = glove_load('glove.6B.100d.txt')

# Load books dataset
books = pd.read_csv('Books_rating.csv')
book_descriptions = books['review/text'].fillna('').tolist()

def get_sentence_embedding(sentence, embeddings_index, embedding_dim=100):
    words = sentence.split()
    valid_words = [word for word in words if word in embeddings_index]
    if not valid_words:
        return np.zeros(embedding_dim)
    embeddings = np.array([embeddings_index[word] for word in valid_words])
    return np.mean(embeddings, axis=0)

# Compute embeddings for book descriptions
book_embeddings = np.array([get_sentence_embedding(description, glove_embedding) for description in book_descriptions])

def get_book_recommendations(input_text, top_k=10):
    input_embedding = get_sentence_embedding(input_text, glove_embedding)
    similarities = cosine_similarity([input_embedding], book_embeddings)[0]
    top_indices = similarities.argsort()[::-1][1:]  # Exclude the first one as it is the query itself
    
    recommendations = []
    seen_titles = set()
    for idx in top_indices:
        book_info = books.iloc[idx]
        title = book_info['Title']
        if title not in seen_titles:
            seen_titles.add(title)
            recommendations.append({
                "Title": title,
                "Score": book_info['review/score'],
                "Price": book_info.get('Price', 'N/A'),
                "Similarity": similarities[idx]
                
            })
        if len(recommendations) >= top_k:
            break
    
    return recommendations
    
    return recommendations

class BookRecommendationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Book Recommendation System")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        self.input_label = QLabel("Enter book name or description:")
        layout.addWidget(self.input_label)
        
        self.input_text = QLineEdit(self)
        layout.addWidget(self.input_text)
        
        self.slider_label = QLabel("Number of Results:")
        layout.addWidget(self.slider_label)
        
        self.result_count = QSlider(Qt.Horizontal)
        self.result_count.setMinimum(1)
        self.result_count.setMaximum(20)
        self.result_count.setValue(10)
        layout.addWidget(self.result_count)
        
        self.button = QPushButton("Show Recommendations", self)
        self.button.clicked.connect(self.show_recommendations)
        layout.addWidget(self.button)
        
        self.table = QTableWidget(self)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Title", "Score", "Price", "Similarity"])
        layout.addWidget(self.table)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def show_recommendations(self):
        input_text = self.input_text.text()
        top_k = self.result_count.value()
        recommendations = get_book_recommendations(input_text, top_k)
        
        self.table.setRowCount(len(recommendations))
        
        for row, rec in enumerate(recommendations):
            self.table.setItem(row, 0, QTableWidgetItem(rec["Title"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{rec['Score']:.1f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"${rec['Price']}" if rec['Price'] != 'N/A' else "N/A"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{rec['Similarity']:.2f}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BookRecommendationApp()
    window.show()
    sys.exit(app.exec())
