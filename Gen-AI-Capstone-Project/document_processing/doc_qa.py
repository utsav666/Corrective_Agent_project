import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import (CharacterTextSplitter,RecursiveCharacterTextSplitter,NLTKTextSplitter,TokenTextSplitter,MarkdownTextSplitter)
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from app_config.open_ai_cred import OPENAI_KEY , TAVILY_API_KEY , WEATHER_API_KEY
from app_config.open_ai_cred import model_name , embed_model
#from doc_indexer import VectorStore
from document_processing.doc_indexer import VectorStore
from document_processing.proj_chains import context_qa_chain,query_re_writer_agent,document_grader_agent
from document_processing.proj_lang_graph import agentic_rag_flow, retriever_call
from langchain_core.messages import HumanMessage
#from langchain_openai import ChatOpenAI
import os 
#import openai

os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
# vector_store = VectorStore(model_name,embed_model)
# chroma_db = vector_store.vectord_db_loader()


# def retriever_call():
#     similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
#                                                         search_kwargs={"k": 3,
#                                                                        "score_threshold": 0.3})
#     final_retriever = similarity_threshold_retriever
#     return final_retriever 

def historical_qa_context(question):
    chatgpt = ChatOpenAI(model_name=model_name,temperature=0)
    final_retriever = retriever_call()
    rag_chain = context_qa_chain(chatgpt,final_retriever)
    result = rag_chain.invoke({"input": question, "chat_history": []})
    print(result)


def agentic_qa(question,chat_history):
    agentic_rag = agentic_rag_flow()
    query = question
    print(len(chat_history),"length of chat hsitory")   
    # if len(chat_history) > 2:
    #     print("greater than 2 so deleted the previous chat")
    #     chat_history = chat_history.pop(0)
    #chat_history  = []
    response = agentic_rag.invoke({"question": query,"chat_history":chat_history })
    
    chat_history.extend([HumanMessage(content=response["question"],response_metadata={"source_document":response["documents"]}), 
                     response["generation"] 
                     #, 
                     #response.get("documents")
                    ])
    #print(chat_history)                            
    return response["generation"] , chat_history



agentic_qa("how does gpt 4 works ?",[])  
#historical_qa_context("how does gpt 4 works ?")
# retrieval_res = retriever_call()
# query = "how does gpt 4 works ?"
# res = retrieval_res.invoke(query)
# # print(res)
# for doc in res:
#     print(doc.page_content)
#     doc_grader  = document_grader_agent()  
#     print('GRADE:', doc_grader.invoke({"question": query, "document": doc.page_content}))
#     print()


  