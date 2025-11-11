import validators,streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()

## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website using Hugging Face Model")
st.subheader('Summarize URL')

## Get the Hugging Face API Key and url(YT or website)to be summarized
with st.sidebar:
    huggingFace_api_key=st.text_input("Hugging Face API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

if not huggingFace_api_key.strip():
    st.error("Please provide the hugging face api key")
    st.stop()

## Model using HuggingFace
repo_id="moonshotai/Kimi-K2-Thinking"
llm=HuggingFaceEndpoint(repo_id=repo_id,max_new_tokens=150,temperature=0.7,huggingfacehub_api_token=huggingFace_api_key)
chat = ChatHuggingFace(llm=llm, verbose=True)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video utl or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                final_documents=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)

                ## Chain For Summarization
                chain=prompt|chat
                output_summary=chain.invoke({'text': final_documents})

                st.success(output_summary.content)
        except Exception as e:
            st.exception(f"Exception:{e}")