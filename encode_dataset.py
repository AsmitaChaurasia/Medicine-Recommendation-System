# encode_dataset.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load dataset
df = pd.read_csv("medicine_dataset.csv")

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + good

# Encode symptoms
embeddings = model.encode(df['symptoms'].tolist())

# Save embeddings and data
with open('medicine_embeddings.pkl', 'wb') as f:
    pickle.dump((df, embeddings), f)
