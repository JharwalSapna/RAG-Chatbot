from conversational_rag import create_context_retriever_chain, create_conversational_rag_chain, create_vectorstore_from_url
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# Set up Streamlit page configuration
st.set_page_config(page_title="Retrieval Augmented Generation", page_icon="💥")
st.title("Natural Language Chat with websites")


# Sidebar Configuration
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


# Function to Get Chatbot Response
def get_response(user_input):
    # Create the retriever chain and conversation RAG chain
    retriever_chain = create_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = create_conversational_rag_chain(retriever_chain)
    
    # Invoke the conversation RAG chain with the user input
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']

# Check if Website URL is Provided
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # Initialize Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_vectorstore_from_url(website_url)    

    # User Input Section
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
