import os
from apikey import apikey
import streamlit as st
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
os.environ ["OPENAI_API_KEY"] = apikey
st.title('Medium Article Generator')
topic = st.text_input('Input your topic of interest')

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Give me medium article title on {topic}'
)

article_template = PromptTemplate(
    input_variables = ['title'],
    template = 'Give me medium article for {title}'
)

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm,prompt=title_template,verbose=True)

llm2 = ChatOpenAI(model_name='gpt-3.5=turbo',temperature=0.9)
article_chain = LLMChain(llm=llm,prompt=article_template,verbose=True)

overall_chain = SimpleSequentialChain(chains=[title_chain,article_chain],verbose=True)

if topic:
    response = overall_chain.run(topic)
    st.write(response)