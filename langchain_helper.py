from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter              #split text into smaller chunks
from langchain_openai import OpenAI , OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS                           # similarity search 

from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import (
    OpenAIWhisperParser,
)
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings()  # In NLP (Natural Language Processing), embedding is the process of representing words or phrases in a high-dimensional numerical vector space. The word or text embeddings are used to capture the underlying meaning and semantic relationships between words in a text corpus. The goal is that similar words are mapped to nearby points, and dissimilar words are mapped to distant points.

def create_vector_db_from_url(video_url) -> FAISS:              # returns a FAISS (Facebook AI Similarity Search) object
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    if len(transcript) == 0 :
        return "No transcript"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_for_query(db,query,k=1):
    docs = db.similarity_search(query,k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model='gpt-3.5-turbo-instruct',temperature = 0)

    prompt = PromptTemplate(
        input_variables=['docs','question'],
        template="""
            You are a helpful Youtube assistant that can answer questions about videos
            based on the videos's transcript.
            Answer the following question : {question}
            By searching the following video transcript : {docs}
            Only use the factual information from the transcript to answer the question.
            If you feel like you don't have enough information to answer the question, say 'I don't know '.
            Your answers should be detailed.
"""
    )
    chain = LLMChain(llm=llm , prompt=prompt)
    response = chain.run(question = query,docs=docs_page_content)
    response = response.replace('\n','')
    return response


def get_audio(urls,save_dir = "C:/Users/abhij/Desktop/LLM Apps/Youtube/audio"):
    loader = GenericLoader(YoutubeAudioLoader([urls], save_dir), OpenAIWhisperParser())
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db


