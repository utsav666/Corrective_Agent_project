import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app_config.open_ai_cred import *
from typing import List
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from app_config.open_ai_cred import OPENAI_KEY , TAVILY_API_KEY , WEATHER_API_KEY
from app_config.open_ai_cred import model_name , embed_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from document_processing.doc_indexer import VectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from document_processing.proj_chains import query_re_writer_agent , context_qa_chain , document_grader_agent
#from doc_qa import retriever_call
from langgraph.graph import END, StateGraph
import os 
os.environ['OPENAI_API_KEY'] = OPENAI_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY
vector_store = VectorStore(model_name,embed_model)
chroma_db = vector_store.vectord_db_loader()

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """

    question: str
    generation: str
    web_search_needed: str
    documents: List[str]
    chat_history: List[str]



def retriever_call():
    similarity_threshold_retriever = chroma_db.as_retriever(search_type="similarity_score_threshold",
                                                        search_kwargs={"k": 3,
                                                                       "score_threshold": 0.3})
    final_retriever = similarity_threshold_retriever
    return final_retriever


#######RETRIEVER##########
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    print("---RETRIEVAL FROM VECTOR DB---")
    question = state["question"]
    chat_history = state['chat_history']
    # Retrieval
    similarity_threshold_retriever = retriever_call()
    documents = similarity_threshold_retriever.invoke(question)
    return {"documents": documents,
            "question": question,
            "chat_history":state['chat_history']}

########GRADER###################
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.

    If any document are not relevant to question or documents are empty - Web Search needs to be done
    If all documents are relevant to question - Web Search is not needed
    Helps filtering out irrelevant documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    doc_grader = document_grader_agent()
    # Score each doc
    filtered_docs = []
    web_search_needed = "No"
    if documents:
        for d in documents:
            score = doc_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search_needed = "Yes"
                continue
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"

    return {"documents": filtered_docs, 
            "question": question, 
            "web_search_needed": web_search_needed,
           "chat_history":state['chat_history']}

##########REWRITER##########################
def rewrite_query(state):
    """
    Rewrite the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased or re-written question
    """

    print("---REWRITE QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    question_rewriter = query_re_writer_agent()
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, 
            "question": better_question,
           "chat_history":state['chat_history']}


from langchain.schema import Document
#############WEBSEARCH######################
def web_search(state):
    """
    Web search based on the re-written question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    #docs = tv_search.invoke(question)
    final_retriever=retriever_call()
    docs = final_retriever.invoke(question)
    #web_results = "\n\n".join([d["content"] for d in docs])
    web_results = "\n\n".join([d.page_content for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, 
            "question": question,
           "chat_history":state['chat_history']}

######GENERATEANSWER######################
def generate_answer(state):
    """
    Generate answer from context document using LLM

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    #generation = qa_rag_chain.invoke({"context": documents, "question": question})
    #chat_history = []
    #print(question,"...question...")
    #print(state['chat_history'],"...chat his....")
    chatgpt = ChatOpenAI(model_name=model_name,temperature=0)
    final_retriever = retriever_call()
    rag_chain = context_qa_chain(chatgpt,final_retriever)
    generation = rag_chain.invoke({"input": question,"chat_history":state['chat_history']})
    
    #chat_history.append(generation)
    return {"documents": documents, 
            "question": question,
            "generation": generation["answer"],
           "chat_history":state['chat_history']}

#########DECISION#################
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    web_search_needed = state["web_search_needed"]

    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REWRITE QUERY---")
        return "rewrite_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE RESPONSE---")
        return "generate_answer"

def agentic_rag_flow():
    agentic_rag = StateGraph(GraphState)

    # Define the nodes
    agentic_rag.add_node("retrieve", retrieve)  # retrieve
    agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
    agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
    agentic_rag.add_node("web_search", web_search)  # web search
    agentic_rag.add_node("generate_answer", generate_answer)  # generate answer

    # Build graph
    agentic_rag.set_entry_point("retrieve")
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)

    # Compile
    agentic_rag = agentic_rag.compile()
    return agentic_rag
