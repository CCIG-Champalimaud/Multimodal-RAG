import streamlit as st
import ast
import string
import shutil
import sys
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from ClassificationWriter import ClassificationWriter, generate_random_id
from EmbeddingFunctions import *
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("🦜🔗 Medical Report RAG App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "context_length" not in st.session_state:
    st.session_state.context_length = 1

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if "rag" not in st.session_state:
    rag, metadata = initialize_multimodal_vector_database()
    st.session_state.rag = rag
    st.session_state.metadata = metadata

#st.write(f"Number of medical reports in the database: {document_collection.count()}")

# Initialize session state for button clicks if not already done
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = None
    st.session_state.output_text = ""

# Function to update session state
def update_output(button_name, output):
    st.session_state.button_clicked = button_name
    st.session_state.output_text = output

if st.text_input('Context Length: '):
    st.session_state.context_length = int(st.text_input('Context Length: '))

#if st.button("Embed Dataset", type="primary"):
#    update_output("Embed Dataset", "Embedding dataset...")

#if st.button("Embed Document", type="primary"):
#    update_output("Embed Document", "Embedding document...")

#if st.button("Delete Report from DB", type="primary"):
#     update_output("Delete Report from DB", "Deleting report from database...")

#if st.button("Delete DB", type="primary"):
#      update_output("Delete DB", "Deleting database...")

# Handle each button separately

#if st.session_state.button_clicked == "Embed Dataset":
#    st.session_state.metadata = add_NLMCXR_to_vectorstore(st.session_state.rag)

    # Print the number of documents in the collection to check if changes were made
#    st.write(f"dataset_micro embedded in the knowledge database! 🚀")
    #st.write(f"Number of medical reports in the database: {document_collection.count()}")

#if st.session_state.button_clicked == "Embed Document":
#    st.write("Feature still not implemented")

#if st.session_state.button_clicked == "Retrieve Report and Image":
#    st.write("Feature still not implemented")
    #retrieved_documents, retrieved_images = retrieve_similar_report(vector_db_text, '/home/ccig/Desktop/Nuno/rag_data/query_test.json', text_collection, vector_db_images, image_collection)
    #img_path = plot_images(retrieved_images)
    #st.write(f"The most similar report that I have access to is the following:")
    #st.write(f"Unique ID: {retrieved_documents[0][0].metadata['uid']}")
    #st.write(f"Content: {retrieved_documents[0][0].page_content}")
    #st.write(f"Found {len(retrieved_images['ids'])} images related to the report")
    #st.write(f"Images shown below:")
    #st.image(img_path, caption="Retrieved Image", use_column_width="auto")

#if st.session_state.button_clicked == "Delete DB":
#    st.write("Feature still not implemented")

#if st.session_state.button_clicked == "Delete Report from DB":
#    st.write("Feature still not implemented")


# React to user input
if message := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": message})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(message)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        llm, processor  = create_generator_model()
        retrieved = retrieve_similar_report(st.session_state.rag, message, st.session_state.context_length)
        response = generate_reply(llm, processor, message, retrieved, st.session_state.metadata)
                    
        st.markdown(response[0])

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
