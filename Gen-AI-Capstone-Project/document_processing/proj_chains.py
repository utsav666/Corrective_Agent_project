import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from app_config.open_ai_cred import OPENAI_KEY , TAVILY_API_KEY , WEATHER_API_KEY
from app_config.open_ai_cred import model_name , embed_model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from document_processing.proj_prompt import contextualize_q_prompt,qa_prompt , re_write_prompt , grade_prompt
import os 
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY


def context_qa_chain(chatgpt,final_retriever):
    history_aware_retriever = create_history_aware_retriever(chatgpt, final_retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(chatgpt, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def query_re_writer_agent():
    llm = ChatOpenAI(model=model_name, temperature=0)
    question_rewriter =(re_write_prompt
                        |
                       llm
                        |
                     StrOutputParser())
    return question_rewriter


def document_grader_agent():
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )
    # LLM for grading
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    doc_grader = (grade_prompt
                  |
              structured_llm_grader)
    return  doc_grader 

# query_writer =    query_re_writer_agent()
# query = "cpitl bharat"
# res = query_writer.invoke({"question": query})
# print(res)
