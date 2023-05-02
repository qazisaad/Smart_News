import asyncio
import aiohttp

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from bs4 import BeautifulSoup
from time import sleep
import streamlit as st
from abc import ABC, abstractmethod
from typing import List, Tuple
from time import time

from Classes.classes import DuckDuckGoNews, Page

import requests

from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
import tiktoken

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

from transformers import BartTokenizer, BartForConditionalGeneration

from scipy.spatial.distance import cosine
from typing import List, Tuple, Union, Any, Dict
import numpy as np
import torch

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

async def process_results(results, max_tokens = 100):
  pages = []
  tasks = []
  async with aiohttp.ClientSession() as session:
    for link, heading, desc in zip(*results):
      page = Page(link, max_tokens)
      task = asyncio.ensure_future(page.parse(True))
      tasks.append(task)
    pages = await asyncio.gather(*tasks)
  return pages

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

@st.cache_resource
def set_embeddings(model_name_dic, model_class_dic, embeddings_type):
  if embeddings_type == 'OpenAI':
    return model_class_dic[embeddings_type]()
  else:
    return model_class_dic[embeddings_type](model_name = model_name_dic[embeddings_type])

def st_wait():
  if query:
    pass
  else:
    st.stop()




#write output using streamlit
def print_without_source(search_results, answer):
  st.write(f'**BOT:**')
  st.write(answer['answer'])
  for id, (link, heading, desc) in enumerate(zip(*search_results)):
    st.write(f'**{id}, {heading}**')
    st.write(desc)
    st.write(link)
    st.write('')

def print_with_source(search_results, answer):
  st.write(f'**BOT:**')
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
  format = None
  if is_source:
    for c in contexts:
        if total_tokens > 3500:
          example_prompts.pop()
          summaries = '\n'.join(example_prompts)
          return summaries
        example = example_prompt.format(page_content = c['text'], source=c['id'])
        example_prompts.append(example)
        total_tokens += llm.get_num_tokens(example)
  else:
      for c in contexts:
        if total_tokens > 3500:
          example_prompts.pop()
          summaries = '\n\n--\n\n'.join(example_prompts)
          return summaries
        example = c['text']
        example_prompts.append(example)
        total_tokens += llm.get_num_tokens(example)
  example_prompts.pop()
  summaries = '\n\n--\n\n'.join(example_prompts)
  return summaries, format

def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['text']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query


@st.cache_resource
def load_bart_model():
    # load bart tokenizer and model from huggingface
    tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
    generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)
    return tokenizer, generator

def bart_generate_answer(query, contexts):
    # format the query and context
    query = format_query(query, contexts)
    # load bart tokenizer and model from huggingface
    tokenizer, generator = load_bart_model()
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt")
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"].to(device), num_beams=2, min_length=20, max_length=40)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    answer = {'answer': answer}
    return answer

def generate_answer_langchain(query, contexts, is_source, example_prompt, max_token_length):
  context = get_summaries(contexts, chat, example_prompt, max_token_length, is_source)

  llm_chain = LLMChain(
    prompt=prompt,
    llm=chat
  )

  with get_openai_callback() as cb:
    answer = llm_chain({'question':query, 'context': context})['text']
    st.write(f'Total cost to generate answer: ${cb.total_cost}')
  if is_source:
    answer = answer.split('SOURCES:')[0]
    source = ''
    if len(answer.split('SOURCES:')) > 1:
      source = answer.split('SOURCES:')[1]
    return {'answer': answer, 'sources': source}
  else:
    return {'answer': answer}

def generate_answer(query, contexts, is_bart):
  if is_bart:
    answer = bart_generate_answer(query, contexts)
  else:
    answer = generate_answer_langchain(query, contexts, is_source, example_prompt, max_tokens_chatbot)
  return answer



def get_top_k_documents_cosine(query: str, texts: List[Any], k: int, is_source: bool, embeddings_function) ->  List[Tuple[float, str, Union[None, Dict[str, str]]]]:
    # Generate the embeddings for the query
    query_embedding = embeddings_function.embed_documents([query])[0]

    # Generate the embeddings for the texts
    text_strings = [doc.page_content for doc in texts]
    text_embeddings = embeddings_function.embed_documents(text_strings)

    # # Calculate cosine similarity between query and text embeddings
    similarities = [1 - cosine(query_embedding, text_embedding) for text_embedding in text_embeddings]

    # Find the top k document indices and their similarities
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = [similarities[idx] for idx in top_k_indices]

    if is_source:
        # Return the texts, similarities, and metadata of the top k documents
        return [{'text': texts[idx].page_content, 'score': similarity, 'id' :texts[idx].metadata['source']} for idx, similarity in zip(top_k_indices, top_k_similarities)]
    else:
        # Return the texts and similarities of the top k documents
        return [{'text': texts[idx].page_content, 'score':similarity, 'id' : None} for idx, similarity in zip(top_k_indices, top_k_similarities)]

def get_top_k_documents_chroma(query: str, texts: List[Any], k: int, is_source: bool, embeddings_function) ->  List[Tuple[float, str, Union[None, Dict[str, str]]]]:
  docsearch = Chroma.from_documents(texts, embeddings_function)
  docs = docsearch.similarity_search_with_score(query, k = k)

  if is_source:
        # Return the texts, similarities, and metadata of the top k documents
        return [{'text': doc[0].page_content, 'score': doc[1], 'id' :doc[0].metadata['source']} for doc in docs]
  else:
      # Return the texts and similarities of the top k documents
      return [{'text': doc[0].page_content, 'score':doc[1], 'id' : None} for doc in docs]



os.environ["OPENAI_API_KEY"] = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write('Device:', device)


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl
QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: Authors name is Jack
Source: 28-pl
Content: Authors Father is Mack
Source: 30-pl
Content: Macks son is Jack
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl, 4-pl
QUESTION: {question}
=========
{context}
=========
FINAL ANSWER:"""
prompt_source = PromptTemplate(template=template, input_variables=["context", "question"])

example_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
Helpful Answer:"""
prompt_no_source = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


t = time()
driver = setup_driver()
print('Time for setting driver', time() - t)

crawler = DuckDuckGoNews(driver)
query = st.text_input('Enter Query:', 'Weather in Norway')
#use streamlit to wait for input before continuing to rest of the code
embeddings_type = st.selectbox(
    'Select Embeddings type',
    ('OpenAI', 'HuggingFace-Mpnet-V2', 'HuggingFace-Mpnet-V3'))


model_name_dic = {'OpenAI': 'text-embedding-ada-002',
'HuggingFace-Mpnet-V2': 'sentence-transformers/all-mpnet-base-v2', 
'HuggingFace-Mpnet-V3': 'flax-sentence-embeddings/all_datasets_v3_mpnet-base'
}

model_class_dic = {'HuggingFace-Mpnet-V2': HuggingFaceEmbeddings, 'OpenAI': OpenAIEmbeddings, 'HuggingFace-Mpnet-V3': HuggingFaceEmbeddings}

embeddings = set_embeddings(model_name_dic, model_class_dic, embeddings_type)


chat = ChatOpenAI(temperature=0)
llm_flan_t5 =HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
# llm_bart =HuggingFaceHub(repo_id="vblagoje/bart_lfqa")
davinci = OpenAI()
generative_model_name_dic = {'Davinvci' : davinci, 'ChatGPT' : chat, 'Flan-t5-xl': llm_flan_t5}
generative_model = st.selectbox(
    'Select Generative model',
    ('Davinvci', 'ChatGPT', 'Flan-t5-xl', 'BART LFQA'))

search_method = st.selectbox(
    'Select Search Method',
    ('ANN', 'Cosine'))

search_method_dic = {'ANN': get_top_k_documents_chroma, 'Cosine': get_top_k_documents_cosine}
search_method_func = search_method_dic[search_method]

is_bart = False
if generative_model != 'BART LFQA':
  llm = generative_model_name_dic[generative_model]
else:
  is_bart = True


# max results using selectbox st for 1 to 10
results_range = (str(i) for i in range(3, 10))
max_results = int(st.selectbox('Select max results', results_range))

tokens_per_chunk = int(st.text_input('Enter tokens per chunk:', '100'))

max_tokens_per_result = int(st.text_input('Enter max tokens per result:', '400'))

max_tokens_chatbot = int(st.text_input('Enter max tokens for chatbot:', '3500'))

top_k = int(st.text_input('Enter top k:', '3'))

top_k = min(top_k, max_results)

is_source = st.checkbox('Print sources?')

prompt = prompt_source if is_source else prompt_no_source

t = time()
search_results = crawler.get_results(query, max_results=max_results)
print('Time for scraping search page', time() - t)
st.write('Time for scraping search page', time() - t)

t = time()
results = asyncio.run(process_results(search_results, max_tokens_per_result))
# results = await results
print('Time for scraping search pages results', time() - t)
st.write('Time for scraping search pages results', time() - t)


t = time()


texts = [x.text for x in  results]

ids = [{'source': i} for i, string in enumerate(texts)]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=tokens_per_chunk, chunk_overlap=20, length_function = tiktoken_len, separators=['\n\n', "\n", ' ', ''])
texts = text_splitter.create_documents(texts, metadatas=ids)
print('Time for splitting text', time() - t)
st.write('Time for splitting text', time() - t)


t = time()

with get_openai_callback() as cb:
  contexts = search_method_func(query, texts, top_k, is_source, embeddings)
  st.write('Time for matching docs', time() - t)
  st.write(f'Total cost for embeddings: ${cb.total_cost}')
print('Time for matching docs', time() - t)

t = time()
answer = generate_answer(query, contexts, is_bart)
print('Time taken to generate answer', time() - t)
st.write('Time taken to generate answer', time() - t)

print_output(search_results, answer, is_source)
# t = time()
# docsearch = Chroma.from_documents(texts, embeddings)
# print('Time for setting up database', time() - t)


# t = time()
# retriever = docsearch.as_retriever()
# print('Time for setting up QA model', time() - t)

# t = time()
# print_output(search_results, query, retriever, is_source)
# print('Time for predictions', time() - t)

# #write output using streamlit
# def print_without_source(search_results, query, qa_model):
#   output = qa_model.run(query)
#   st.write(f'**BOT:**')
#   st.write(output)
#   for id, (link, heading, desc) in enumerate(zip(*search_results)):
#     st.write(f'**{id}, {heading}**')
#     st.write(desc)
#     st.write(link)
#     st.write('')

# def print_with_source(search_results, query, qa_model):
#   output = qa_model({'question': query}, return_only_outputs=True)
#   st.write(f'**BOT:**')
#   st.write(output['answer'])
#   st.write('**SOURCES**')
#   st.write(output['sources'])
#   for id, (link, heading, desc) in enumerate(zip(*search_results)):
#     st.write(f'**{id}, {heading}**')
#     st.write(desc)
#     st.write(link)
#     st.write('')

# def print_output(search_results, query, retriever, is_source):
#   if is_source:
#     qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
#     print_with_source(search_results, query, qa)
#   else:
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
#     print_without_source(search_results, query, qa)