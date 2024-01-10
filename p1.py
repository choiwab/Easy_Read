__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import streamlit as st
import time
import os

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities.dalle_image_generator import DallEAPIWrapper

from bs4 import BeautifulSoup
import validators  # To validate URL
import requests
from pypdf import PdfReader 
from fpdf import FPDF
from io import BytesIO
from PIL import Image


# import nltk
# #nltk.download()
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize

llm = OpenAI(temperature=0.1)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Create a concise prompt for an textless image that illustrates '{image_desc}'. Include in the prompt to not contain text or writing of any kind.",
)
chain = LLMChain(llm=llm, prompt=prompt)


#로컬 환경에서 내 api key로 돌릴때 
# ---------------------------------------------------
#os.environ["OPENAI_API_KEY"] ="" 
# ---------------------------------------------------

#첫번째 구현 방법: Streamlit 배포할때 OpenAI API key로 돌려도 된다면 다음 코드로 배포하기
#대신 streamlit에서 따로 api key를 추가해야합니다.
#---------------------------------------------------
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#---------------------------------------------------

# 두번째 구현 방법: 사용자의 api key 받아서 돌리기
# ---------------------------------------------------
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Enter OPENAI API Key")
    st.stop()

import os
os.environ["OPENAI_API_KEY"] = openai_api_key
# ---------------------------------------------------


# temperature는 0에 가까워질수록 형식적인 답변을 내뱉고, 1에 가까워질수록 창의적인 답변을 내뱉음
llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-1106")

# 어떤 파일을 학습시키는지에 따라 코드를 바꿔주세요. ex) pdf, html, csv

# 첫번째 구현 방법: 웹사이트 url 학습시키기
# ---------------------------------------------------
# from langchain.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://sosoeasyword.com/27/?q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&bmode=view&idx=17122350&t=board")
# data = loader.load()
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
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path= 'instructions.csv')
data = loader.load()
print(data)
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
       # "You are service agent that converts complex reading material into easy read material for mentally disabled people"
        "You are service agent for mentally disabled people"
        "If you are given a complex reading material, you will convert it easy read material."
        "If you are asked a question, you will answer in a simple manner."
        "Do your best to convert the reading material into the most simple terms"
        "Look up for the example using the tools you have"
        "Follow these rules: "  
        "1. Write in short sentences of 15-20 words"
        "2. Skip a line for each sentence"
        "3. Write as if you are speaking."
        "4. Each sentence has one idea."
        "5. Use active verbs as much as possible."
        "6. Keep the language personal e.g. you, we "
        "7. Use drop down bullet points to list."
        "8. Reduce punctuation as much as you can."
       #"Make sure to answer in Korean."
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

# Function to scrape text from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return ' '.join(p.get_text().strip() for p in soup.find_all('p'))
        else:
            return "Failed to retrieve content from URL."
    except Exception as e:
        return f"An error occurred: {e}"

# from langchain.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://sosoeasyword.com/27/?q=YToxOntzOjEyOiJrZXl3b3JkX3R5cGUiO3M6MzoiYWxsIjt9&bmode=view&idx=17122350&t=board")
# data = loader.load()

# Function to process the user input
def process_user_input(user_input):
    if validators.url(user_input):  # Check if input is a valid URL
        return scrape_text_from_url(user_input)
    # elif user_input.endswith('.pdf'):  # Check if input is a PDF file
    #     return extract_text_from_pdf(user_input)
    else:
        return user_input  # Treat input as plain text
    
def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)
        pdf_text = ''
        for page in reader.pages:
            pdf_text += page.extract_text()
    return pdf_text

def create_pdf(text, image_url):
    pdf = FPDF()
    pdf.add_page()

    # Add text
    # Add a Unicode-compatible font
    # Make sure the DejaVuSansCondensed.ttf file is in your project directory or provide the correct path
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', size=15)
    pdf.multi_cell(0, 10, text)

    # Add image
    image_path = download_image(image_url)
    pdf.image(image_path, x=10, y=pdf.get_y(), w=200)  # Adjust dimensions as needed

    pdf_output = "output.pdf"
    pdf.output(pdf_output)
    return pdf_output


def download_image(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image_path = 'temp_image.jpg'  # Temporary file path
    image.save(image_path)
    return image_path


# def format_response(text):
#     sentences = sent_tokenize(text)
#     formatted_response = '\n\n'.join(sentences)
#     return formatted_response

# 웹사이트 제목
st.title("Easy Read Generator📖")
st.markdown("⭐️Ask Me Anything / Copy & Paste Difficult Text / Copy & Paste URL⭐️")
st.markdown("🌟Easy Read Material for Everyone😌")

st.image('easyread.jpeg') 

with st.expander('What Is Easy Read?'):
    st.write('‘Easy read’ refers to the presentation of text in an accessible, easy to understand format. It is often useful for people with learning disabilities, and may also be beneficial for people with other conditions affecting how they process information. \n\n More Info Here: https://www.learningdisabilities.org.uk/learning-disabilities/a-to-z/e/easy-read')
with st.expander('What Is Learning Disability?'):
    st.write("A learning disability, also known as a learning disorder, is a neurological condition that affects an individual's ability to acquire, process, store, and use information effectively. These disabilities can manifest in various ways and can impact an individual's performance in one or more areas of learning, such as reading, writing, mathematics, or problem-solving. Learning disabilities are typically present from childhood and often persist into adulthood. \n\n Find out more here: https://www.mencap.org.uk/learning-disability-explained/what-learning-disability#:~:text=A%20learning%20disability%20is%20to,someone%20for%20their%20whole%20life.")
with st.expander('How Many People Have Learning Disabilities?'):
    st.write('At least 1 in every 59 children has one or several learning disabilities and 1 in 5 children in the U.S. have learning and thinking differences such as ADHD or Dyslexia')
with st.expander('How Can I Use this Website?'):
    st.write('This platform is designed for both educational purposes and to assist individuals with learning disabilities in simplifying essential information for their everyday needs.')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# pdf_text = None 
# if uploaded_file is not None: 
#     # Save the file locally
#     with open("temp_pdf_file.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     pdf_text = extract_text_from_pdf("temp_pdf_file.pdf")
        
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
pdf_text = None 

if uploaded_file is not None: 
    # Use a PDF processing library to read from the file-like object
    try:
        reader = PdfReader(uploaded_file)
        pdf_text = ''
        for page in reader.pages:
            pdf_text += page.extract_text() + '\n'
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")

# if 'last_processed_input' not in st.session_state:
#     st.session_state.last_processed_input = None
#     st.session_state.generated_text = None
#     st.session_state.generated_image_url = None

# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# if uploaded_file is not None:
#     # Read the uploaded file
#     pdf_text = extract_text_from_pdf(uploaded_file)

#     # Check if the input has changed
#     if pdf_text != st.session_state.last_processed_input:
#         # Process the new input
#         processed_text, image_url = process_input_with_llm(pdf_text) 
        
#         # Update session state
#         st.session_state.last_processed_input = pdf_text
#         st.session_state.generated_text = processed_text
#         st.session_state.generated_image_url = image_url


user_input = st.chat_input("Enter text/URL")

if user_input:
    prompt = user_input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
elif pdf_text:
    prompt = pdf_text
else:
    prompt = None

full_response = ""
image_url = ""

if prompt:
     prompt = process_user_input(prompt)

# AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
     with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        result = agent_executor({"input": prompt})

        for chunk in result["output"].split():
            #full_response += chunk + " "
            full_response += chunk.replace('.', '.\n\n') + " " 
            time.sleep(0.1) 
            message_placeholder.markdown(full_response + "▌") 
        message_placeholder.markdown(full_response) 
        #      full_response += chunk + " "
        #      formatted_response = format_response(full_response)
        #      time.sleep(0.1)
        #      message_placeholder.markdown(formatted_response + "▌")
        # message_placeholder.markdown(formatted_response)
        image_prompt = chain.run(result["output"])
        print(result["output"])
        print(image_prompt)
        image_url = DallEAPIWrapper().run(image_prompt)
        print(image_prompt)
        st.image(image_url)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        pdf_file = create_pdf(full_response, image_url) # Replace with your image path 

        with open(pdf_file, "rb") as file:
            st.download_button(
            label="Download PDF",
            data=file,
            file_name="easy_read_output.pdf",
            mime="application/octet-stream"
    )
 
