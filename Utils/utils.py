import asyncio
import os
from abc import ABC, abstractmethod
from time import sleep, time
from typing import List, Tuple, Union, Any, Dict

import aiohttp
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from langchain import LLMChain
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import Chroma
from scipy.spatial.distance import cosine
from selenium import webdriver
from transformers import BartTokenizer, BartForConditionalGeneration



@st.cache_resource
def setup_driver():
  options = webdriver.ChromeOptions()
  # run Selenium in headless mode
  options.add_argument('--headless')
  options.add_argument('--no-sandbox')
  # overcome limited resource problems
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument("lang=en")
  # open Browser in maximized mode
  options.add_argument("start-maximized")
  # disable infobars
  options.add_argument("disable-infobars")
  # disable extension
  options.add_argument("--disable-extensions")
  options.add_argument("--incognito")
  options.add_argument("--disable-blink-features=AutomationControlled")
  driver = webdriver.Chrome(options=options)
  return driver





#write output using streamlit
def print_without_source(search_results, answer):
#   st.write(f'**BOT:**')
  st.write(answer['answer'])
  for id, (link, heading, desc) in enumerate(zip(*search_results)):
    st.write(f'**{id}, {heading}**')
    st.write(desc)
    st.write(link)
    st.write('')

def print_with_source(search_results, answer):
#   st.write(f'**BOT:**')
  st.write(answer['answer'])
  st.write('**SOURCES**')
  st.write(answer['sources'])
  for id, (link, heading, desc) in enumerate(zip(*search_results)):
    st.write(f'**{id}, {heading}**')
    st.write(desc)
    st.write(link)
    st.write('')

def print_output(search_results, answer, is_source):
  if is_source:
    print_with_source(search_results, answer)
  else:
    print_without_source(search_results, answer)

def get_summaries(contexts, llm, example_prompt, max_token_length, is_source):
    example_prompts = []
    total_tokens = 0
    max_token_length = min(3500, max_token_length)
    if is_source:
        for c in contexts:
            if total_tokens > max_token_length:
                break
            example = example_prompt.format(page_content=c["text"], source=c["id"])
            example_tokens = llm.get_num_tokens(example)
            if total_tokens + example_tokens > max_token_length:
                break
            total_tokens += example_tokens
            example_prompts.append(example)
    else:
        for c in contexts:
            if total_tokens > max_token_length:
                break
            example_tokens = llm.get_num_tokens(c["text"])
            if total_tokens + example_tokens > max_token_length:
                break
            total_tokens += example_tokens
            example_prompts.append(c["text"])
    summaries = "\n\n--\n\n".join(example_prompts)
    return summaries


def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['text']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query







def get_top_k_documents(query: str, texts: List[Any], k: int, is_source: bool, method: str, embeddings_function) -> List[Tuple[float, str, Union[None, Dict[str, str]]]]:
   if method == 'Cosine':
    # Generate the embeddings for the query
    query_embedding = embeddings_function.embed_documents([query])[0]
     # Generate the embeddings for the texts
    text_strings = [doc.page_content for doc in texts]
    text_embeddings = embeddings_function.embed_documents(text_strings)

    # Calculate cosine similarity between query and text embeddings
    similarities = [1 - cosine(query_embedding, text_embedding) for text_embedding in text_embeddings]

    # Find the top k document indices and their similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = [similarities[idx] for idx in top_k_indices]

    top_pages = [texts[idx].page_content for idx in top_k_indices]
    if is_source:
        top_sources = [texts[idx].metadata['source'] for idx in top_k_indices]
    else:
       top_sources = [None for idx in top_k_indices]
   elif method == 'ANN':
    docsearch = Chroma.from_documents(texts, embeddings_function)
    docs = docsearch.similarity_search_with_score(query, k=k)

    # Find the top k document indices and their similarities
    top_k_indices = [i for i in range(len(texts))]
    top_k_similarities = [doc[1] for doc in docs]

    top_pages  = [doc[0].page_content for doc in docs]
    if is_source:
        top_sources = [doc[0].metadata['source'] for doc in docs]
    else:
         top_sources = [None for doc in docs]
   else:
    raise ValueError("Invalid method provided. Use 'Cosine' or 'ANN'.")

   if is_source:
    # Return the texts, similarities, and metadata of the top k documents
    return [{'text': page_content, 'score': similarity, 'id': source} for page_content, similarity, source in zip(top_pages, top_k_similarities, top_sources)]
   else:
    # Return the texts and similarities of the top k documents
    return [{'text': page_content, 'score': similarity, 'id': source} for page_content, similarity, source in zip(top_pages, top_k_similarities, top_sources)]