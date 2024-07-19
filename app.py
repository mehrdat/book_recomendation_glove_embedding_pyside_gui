import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import dearpygui.dearpygui as dpg
from sklearn.metrics.pairwise import cosine_similarity

# with open('glove.6B.100d.txt') as f:
#     text = f.read()

#print(text[:100])

def glove_load(file_path):
    embeddings_index={}
    with open(file_path) as f:
        for line in f:
            values = line.split()
            word=values[0]
            coefs=np.asarray(values[1:],dtype='float32')
            embeddings_index[word]=coefs                
    return embeddings_index

glove_embedding=glove_load('glove.6B.100d.txt')
#print('Found %s word vectors.' % len(glove_embedding))


books = pd.read_csv('Books_rating_10000.csv')

books=books.drop_duplicates(subset='review/text', keep='first')
books=books.dropna(axis=0, subset=['review/text'])
book_transcription=books['review/text'].tolist()
#book_transcriptions = books['review/text'].fillna('').tolist()


def get_sentence_embedding(sentence,embeddings_index):
    words=sentence.split()
    word_vectors=[]
    valid_embedding=[embeddings_index[word] for word in words if word in embeddings_index]
    if not valid_embedding:
        return np.zeros(300)
    return np.mean(valid_embedding,axis=0)

#embeddings=[get_sentence_embedding(sentence,glove_embedding) for sentence in book_transcription]
book_embeddings = np.array([get_sentence_embedding(description, glove_embedding) for description in book_descriptions])

#loaded_embeddings = np.load('embeddings.npy')

embeddings=pd.read_csv('embeddings.csv')
# def get_book_recommendation(book_index,top_k=5):
#     book_embeding=embeddings[book_index]
#     similarity=np.dot(embeddings,book_embeding)/(np.linalg.norm(embeddings,axis=1)*np.linalg.norm(book_embeding))
#     most_similar_books=np.argsort(similarity)[::-1][1:top_k+1]
    
#     recommendations=[]
#     for books in most_similar_books:
#         recommendations.append(books)
#     return recommendations


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

with dpg.window(label="Book Recommendation System"):
    dpg.add_text("Enter book name or description:")
    dpg.add_input_text(label="", tag="Input_Text")
    dpg.add_button(label="Show Recommendations", callback=show_recommendations)
    dpg.add_separator()
    dpg.add_text("Recommendations:")
    dpg.add_text("", tag="Recommendation_Text")

dpg.create_viewport(title='Book Recommendation System', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()





# def show_recommendations(sender, data):
#     book_index = dpg.get_value("Book Index")
#     recommendations = get_book_recommendations(book_index)
#     dpg.set_value("Recommendations", recommendations)

# dpg.create_context()

# with dpg.window(label='Book Recommendation System'):
#     dpg.add_input_text(label='Book Index', default_value='0')
#     # dpg.add_button(label='Show Recommendations', callback=show_recommendations)
#     # dpg.add_text(label='Recommendations', default_value='')
#     for i,book in books.iterrows():
#         with dpg.group(horizontal=True):
#             dpg.add_button(label=book['Title'], callback=show_recommendations,user_data=i)
#             #print(book)
#     dpg.add_separator()
#     dpg.add_text(label='Recommendations:')
#     dpg.add_text("",tag="Recommendation_text")

# dpg.create_viewport(title='Book Recommendation System', width=800, height=800)
# dpg.setup_dearpygui()
# dpg.show_viewport()
# dpg.start_dearpygui()
# dpg.destroy_context()



