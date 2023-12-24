# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")
#print(openai.api_key)

#로컬 환경에서 내 api key로 돌릴때 
# ---------------------------------------------------
os.environ["OPENAI_API_KEY"] = "openai.api_key"
# ---------------------------------------------------

#첫번째 구현 방법: Streamlit 배포할때 OpenAI API key로 돌려도 된다면 다음 코드로 배포하기
#대신 streamlit에서 따로 api key를 추가해야합니다.
#---------------------------------------------------
#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#---------------------------------------------------

# 두번째 구현 방법: 사용자의 api key 받아서 돌리기
# ---------------------------------------------------
# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# if not openai_api_key:
#     st.info("OpenAI API를 먼저 입력해주세요.")
#     st.stop()

# import os
# os.environ["OPENAI_API_KEY"] = openai_api_key
# ---------------------------------------------------


# temperature는 0에 가까워질수록 형식적인 답변을 내뱉고, 1에 가까워질수록 창의적인 답변을 내뱉음
llm = ChatOpenAI(temperature=0.2)

# 어떤 파일을 학습시키는지에 따라 코드를 바꿔주세요. ex) pdf, html, csv

# 첫번째 구현 방법: 웹사이트 url 학습시키기
# ---------------------------------------------------
from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://dalpha.so/ko/howtouse?scrollTo=custom")
data = loader.load()
# ---------------------------------------------------


# 두번째 구현 방법: pdf 학습시키기
# 먼저 VSCode에서 만든 이 폴더 내에 pdf 파일을 업로드 해주셔야해요!
# 사용하고 싶으면 아래 부분의 코드 주석을 없애주세요
# ---------------------------------------------------
# from langchain.document_loaders import PyPDFLoader

# loader = PyPDFLoader("파일이름.pdf")
# pages = loader.load_and_split()

# data = []
# for content in pages:
#     data.append(content)
# ---------------------------------------------------


# 세번째 구현 방법: csv 학습시키기
# 먼저 VSCode에서 만든 이 폴더 내에 csv 파일을 업로드 해주셔야해요!
# 사용하고 싶으면 아래 부분의 코드 주석을 없애주세요
# ---------------------------------------------------
# from langchain.document_loaders.csv_loader import CSVLoader

# loader = CSVLoader(file_path='파일이름.csv')
# data = loader.load()
# ---------------------------------------------------

# 올린 파일 내용 쪼개기
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

# 쪼갠 내용 vectorstore 데이터베이스에 업로드하기
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 데이터베이스에 업로드 한 내용을 불러올 수 있도록 셋업
retriever = vectorstore.as_retriever()

# 에이전트가 사용할 내용 불러오는 툴 만들기
from langchain.agents.agent_toolkits import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "cusomter_service",
    "Searches and returns documents regarding the customer service guide.",
)
tools = [tool]

# 대화 내용 기록하는 메모리 변수 셋업
memory_key = "history"

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder

# AI 에이전트가 사용할 프롬프트 짜주기
system_message = SystemMessage(
    content=(
        "You are a nice customer service agent."
        "Do your best to answer the questions."
        "Feel free to use any tools available to look up "
        "relevant information, only if necessary"
        "Do not generate false answers to questions that are not related to the customer service guide."
        "If you don't know the answer, just say that you don't know. Don't try to make up an answer."
        "Make sure to answer in Korean."
    )
)

prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=system_message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
)

# 에이전트 셋업해주기
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

from langchain.agents import AgentExecutor

# 위에서 만든 툴, 프롬프트를 토대로 에이전트 실행시켜주기 위해 셋업
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    return_intermediate_steps=True,
)

# 웹사이트 제목
st.title("AI 상담원")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("Dalpha AI store는 어떻게 사용하나요?"):

# 유저가 보낸 질문이면 유저 아이콘과 질문 보여주기
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = agent_executor({"input": prompt})
        for chunk in result["output"].split():
            full_response += chunk + " "
            time.sleep(0.1)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})