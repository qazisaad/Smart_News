import torch
from datasets import Dataset, load_dataset
from huggingface_hub.hf_api import HfFolder 
from transformers import AutoTokenizer, AutoModel
import numpy as np
import streamlit as st
import pandas as pd



HfFolder.save_token('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')

def vectorize_query(text):
  inputs = tokenizer(text, truncation=True , return_tensors="pt", padding = True)
  for key, value in inputs.items():
    inputs[key] = value.to(device)
  embeddings = ce(**inputs)[1][0].cpu().detach().numpy()
  return embeddings

@st.cache_resource
def load_model():
  tokenizer = AutoTokenizer.from_pretrained('qazisaad/MiniLM-L6-v2-news-recommender')
  device = "cuda" if torch.cuda.is_available() else "cpu"
  ce = AutoModel.from_pretrained('qazisaad/MiniLM-L6-v2-news-recommender')
  ce.to(device)
  return ce, tokenizer, device

@st.cache_data
def load_data():
  ds = load_dataset('qazisaad/news_recommendations_base_vectorized', split='train')
  ds.add_faiss_index(column='embeddings')
  return ds

ce, tokenizer, device = load_model()
ds = load_data()


categories = list(np.unique(ds['category'][:]))
options = st.multiselect(
    'Select your favorite categories',
    categories,
    [])


question = st.text_input('Enter user interests seperated by "->" character: ', '').replace('->', ' [SEP] ')
question += " [SEP] ".join(options)
question_embedding = vectorize_query(question)

k = int(st.text_input('Enter Top K News to Recommend', '5'))
scores, retrieved_examples = ds.get_nearest_examples('embeddings', question_embedding, k=k)

st.write('Top K News is: ')
del retrieved_examples['embeddings']
del retrieved_examples['category']
del retrieved_examples['sub-category']

df  = pd.DataFrame(retrieved_examples)
st.table(df)

