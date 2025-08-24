# recommend.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load BERT model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('medicine_embeddings.pkl', 'rb') as f:
    df, embeddings = pickle.load(f)

def recommend_top_medicines(user_input, top_k=3):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    result_df = df.iloc[top_indices].copy()
    result_df['similarity_score'] = [round(similarities[i], 3) for i in top_indices]
    result_df.reset_index(drop=True, inplace=True)
    return result_df
