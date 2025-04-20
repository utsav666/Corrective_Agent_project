import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from datetime import datetime
from document_processing.doc_qa import agentic_qa
# global conv_history
# conv_history = []

# Session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conv_history" not in st.session_state:
    st.session_state.conv_history = []




def chatbot_response(user_input):
    """Generate a simple response based on user input."""
    question =user_input
    answer , conv = agentic_qa(question, st.session_state.conv_history)
    #st.session_state.conv_history.extend(conv)
    return answer

# Streamlit UI
st.title("Chatbot with Streamlit")




# Input text box for user query
def submit_chat():
    user_input = st.session_state.user_input
    if user_input.strip():
        # Get response from the chatbot
        print(len(st.session_state.conv_history ), "....before chat...")
        bot_response = chatbot_response(user_input)
        print(len(st.session_state.conv_history) , "....after chat...")
        # Append user input and bot response to chat history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_response))
        st.session_state.user_input = ""  # Clear the input box

# Attach the function to session state
st.session_state.submit_chat = submit_chat

# Create a scrollable text box to display chat
chat_box_style = """
<div style="height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">
"""
chat_box_content = ""
for speaker, message in st.session_state.chat_history:
    if speaker == "You":
        chat_box_content += f"<p><strong>{speaker}:</strong> {message}</p>"
    else:
        chat_box_content += f"<p>{speaker}: {message}</p>"
chat_box_style += chat_box_content + "</div>"

st.markdown(chat_box_style, unsafe_allow_html=True)

# Input box at the bottom for user queries
user_input = st.text_input("", "", key="user_input", on_change=submit_chat, placeholder="Type your message and press Enter...")
