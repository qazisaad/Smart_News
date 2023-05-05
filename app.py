import asyncio
import os
from abc import ABC, abstractmethod
from time import sleep, time
from typing import List, Tuple, Union, Any, Dict

import aiohttp
import numpy as np
import streamlit as st
import torch
import tiktoken
from bs4 import BeautifulSoup
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from scipy.spatial.distance import cosine
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from transformers import BartTokenizer, BartForConditionalGeneration


from Utils.classes import DuckDuckGoNews, Page
from Utils.utils import *
from Utils.prompts import *


class StreamingStdOutCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming. Only works with LLMs that support streaming."""
    # initilize init with super class
    def __init__(self) -> None:
        super().__init__()
        self.text = ''

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        # st.write(token)
        self.text = self.text + token
        # output_container.text(self.text)
        output_container.markdown(f'{self.text}') 

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Run when tool starts running."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> None:
        """Run on arbitrary text."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""


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
  context = get_summaries(contexts, llm, example_prompt, max_token_length, is_source)

  llm_chain = LLMChain(
    prompt=prompt,
    llm=llm
  )


  answer_resonse = llm_chain({'question':query, 'context': context})['text']
  total_tokens = len(tokenizer.encode(llm_chain.prompt.template)) + len(tokenizer.encode(query)) + len(tokenizer.encode(context))
  cost = total_tokens * model_price[generative_model] if generative_model in model_price.keys() else 0
  st.write(f'Total cost to generate answer: ${cost}')
  
  if is_source:
    answer = answer_resonse.split('SOURCES:')[0]
    source = ''
    if len(answer_resonse.split('SOURCES:')) > 1:
      source = answer_resonse.split('SOURCES:')[1]
    return {'answer': answer, 'sources': source}
  else:
    return {'answer': answer_resonse}

def generate_answer(query, contexts, is_bart):
  if is_bart:
    answer = bart_generate_answer(query, contexts)
  else:
    answer = generate_answer_langchain(query, contexts, is_source, example_prompt, max_tokens_chatbot)
  return answer

tokenizer = tiktoken.get_encoding('cl100k_base')




device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.write('Device:', device)


os.environ["OPENAI_API_KEY"] = st.text_input('Enter OpenAI API Key:', '')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.text_input('Enter HuggingFaceHub API Token:', '')

start_time = time()
driver = setup_driver()
print('Time for setting driver', time() - start_time)

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


chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
llm_flan_t5 =HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
# llm_bart =HuggingFaceHub(repo_id="vblagoje/bart_lfqa")
davinci = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)

generative_model_name_dic = {'Davinvci' : davinci, 'ChatGPT' : chat, 'Flan-t5-xl': llm_flan_t5}
generative_model = st.selectbox(
    'Select Generative model',
    ('Davinvci', 'ChatGPT', 'Flan-t5-xl', 'BART LFQA'))

search_method = st.selectbox(
    'Select Search Method',
    ('ANN', 'Cosine'))

model_price =  {'ChatGPT': 0.002/1000, 'Davinvci': 0.1200/1000, 'OpenAI': 0.0004/1000}

is_bart = False
if generative_model != 'BART LFQA':
  llm = generative_model_name_dic[generative_model]
else:
  is_bart = True


# max results using selectbox st for 1 to 10
results_range = (f"{i}" for i in range(3, 10))
max_results = int(st.selectbox('Select max results', results_range))

tokens_per_chunk = int(st.text_input('Enter tokens per chunk:', '100'))

max_tokens_per_result = int(st.text_input('Enter max tokens per result:', '400'))

max_tokens_chatbot = int(st.text_input('Enter max tokens for chatbot:', '3500'))

top_k = int(st.text_input('Enter top k:', '3'))

top_k = min(top_k, max_results)

is_source = st.checkbox('Print sources?')

prompt = prompt_source if is_source else prompt_no_source

start_time = time()
search_results = crawler.get_results(query, max_results=max_results)
print('Time for scraping search page', time() - start_time)
st.write('Time for scraping search page', time() - start_time)

start_time = time()
results = asyncio.run(process_results(search_results, max_tokens_per_result))
# results = await results
print('Time for scraping search pages results', time() - start_time)
st.write('Time for scraping search pages results', time() - start_time)


start_time = time()


texts = [x.text for x in  results]

ids = [{'source': i} for i, string in enumerate(texts)]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=tokens_per_chunk, chunk_overlap=20, length_function = tiktoken_len, separators=['\n\n', "\n", ' ', ''])
texts = text_splitter.create_documents(texts, metadatas=ids)
print('Time for splitting text', time() - start_time)
st.write('Time for splitting text', time() - start_time)


start_time = time()


contexts = get_top_k_documents(query, texts, top_k, is_source, search_method, embeddings)
st.write('Time for matching docs', time() - start_time)
total_tokens = len(tokenizer.encode(query))
total_tokens += sum([len(tokenizer.encode(text.page_content)) for text in texts])
cost = total_tokens * model_price[embeddings_type] if embeddings_type in model_price.keys() else 0
st.write(f'Total cost for embeddings: ${cost}')
print('Time for matching docs', time() - start_time)

start_time = time()

st.write(f'**BOT:**')
output_container = st.empty()

answer = generate_answer(query, contexts, is_bart)
print('Time taken to generate answer', time() - start_time)
st.write('Time taken to generate answer', time() - start_time)

print_output(search_results, answer, is_source)