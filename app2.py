import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dearpygui.dearpygui as dpg
from sklearn.metrics.pairwise import cosine_similarity

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
books = pd.read_csv('Books_rating_10000.csv')
book_descriptions = books['review/text'].fillna('').tolist()

def get_sentence_embedding(sentence, embeddings_index):
    words = sentence.split()
    valid_words = [word for word in words if word in embeddings_index]
    if not valid_words:
        return np.zeros(100)
    embeddings = np.array([embeddings_index[word] for word in valid_words])
    return np.mean(embeddings, axis=0)

# Compute embeddings for book descriptions
book_embeddings = np.array([get_sentence_embedding(description, glove_embedding) for description in book_descriptions])





def get_book_recommendations(input_text, top_k=10):
    input_embedding = get_sentence_embedding(input_text, glove_embedding)
    similarities = cosine_similarity([input_embedding], book_embeddings)[0]
    top_indices = similarities.argsort()[::-1][1:top_k + 1]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append((books.iloc[idx]['Title'], similarities[idx]))
    
    return recommendations




def show_recommendations(sender, app_data, user_data):
    input_text = dpg.get_value("Input_Text")
    recommendations = get_book_recommendations(input_text)
    recommendations_text = "\n".join([f"{title} (Score: {score:.2f})" for title, score in recommendations])
    dpg.set_value("Recommendation_Text", recommendations_text)

dpg.create_context()

with dpg.window(label="Book Recommendation System", width=800, height=600):
    with dpg.group(horizontal=False):
        dpg.add_text("Enter book name or description:")
        dpg.add_input_text(label="", tag="Input_Text", width=400)
        dpg.add_button(label="Show Recommendations", callback=show_recommendations)
        dpg.add_separator()
        dpg.add_text("Recommendations:")
        with dpg.child_window(tag="Recommendation_Window", width=750, height=400, autosize_x=True, border=True):
            dpg.add_text("", tag="Recommendation_Text", wrap=700)

dpg.create_viewport(title='Book Recommendation System', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()