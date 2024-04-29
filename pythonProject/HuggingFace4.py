import streamlit as st
from langchain import PromptTemplate, HuggingFaceHub
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import os

#input your hf token here
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''

st.title("Blog Generator with LangChain")

prompt = st.text_input("Type your content that you want to post ")

#template for both title and script
title_template = """ content : {topic}
Answer: Write me Blog post title. Title start with Discovering the Secrets of  {topic}. """

script_template= """ content : {title}
Answer: Write me  Blog content based on this TITLE {title} . The article start with  Unraveling the Mysteries of {topic}"""

# Define prompt template
title_template = PromptTemplate(
    input_variables=['topic'],
    template=title_template
)

script_template = PromptTemplate(
    input_variables=['title'],
    template=script_template
)

#save to memory
title_memory=ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory=ConversationBufferMemory(input_key='title', memory_key='chat_history')

#create llm
llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={'temperature':0, 'max_length':264})
#chain tgt
title_chain=LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain=LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

sequential_chain=SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'],
                                 output_variables=['title', 'script'], verbose=True)

if prompt:
    response= sequential_chain.invoke({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)
