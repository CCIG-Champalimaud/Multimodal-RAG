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

st.title("🦜🔗 RAG App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Define a template for the chat
template = """Use only the given context to answer the question. 
You can not use any knowledge that isn't present in the context.
If you can't answer the question using only the provided context say 'I don't have enough context to answer the question'". 
Don't hallucinate.
context: {context}
question: {question} according to the provided context?

Answer:"""
prompt = ChatPromptTemplate.from_template(template)


#Initialize Database and LLM
vector_db_text, vector_db_images, text_collection, image_collection = initialize_multimodal_vector_database("vector_db", "vector_db")
llm = get_chat_model('ollama', 'llama3')

st.write(f"Number of medical reports in the database: {text_collection.count()}")
st.write(f"Number of medical images in the database: {image_collection.count()}")

# Initialize session state for button clicks if not already done
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = None
    st.session_state.output_text = ""

# Function to update session state
def update_output(button_name, output):
    st.session_state.button_clicked = button_name
    st.session_state.output_text = output

if st.button("Embed Dataset", type="primary"):
    update_output("Embed Dataset", "Embedding dataset...")

if st.button("Embed Document", type="primary"):
    update_output("Embed Document", "Embedding document...")


if st.button("Retrieve Report", type="primary"):
    update_output("Retrieve Report", "Retrieving report...")

if st.button("Retrieve Report and Image", type="primary"):
    update_output("Retrieve Report and Image", "Retrieving report and image...")

if st.button("Delete Report from DB", type="primary"):
     update_output("Delete Report from DB", "Deleting report from database...")

if st.button("Delete DB", type="primary"):
      update_output("Delete DB", "Deleting database...")

# Handle each button separately

if st.session_state.button_clicked == "Embed Dataset":
    add_NLMCXR_to_vectorstore('/home/ccig/Desktop/Nuno/rag_data/dataset_micro.json', text_collection, image_collection)

    # Print the number of documents in the collection to check if changes were made
    st.write(f"dataset_micro embedded in the knowledge database! 🚀")
    st.write(f"Number of medical reports in the database: {text_collection.count()}")
    st.write(f"Number of medical images in the database: {image_collection.count()}")

if st.session_state.button_clicked == "Embed Document":
    st.write("Feature still not implemented")

if st.session_state.button_clicked == "Retrieve Report":
    retrieved_documents = retrieve_similar_report(vector_db_text, '/home/ccig/Desktop/Nuno/rag_data/query_test.json', text_collection)
    st.write(f"The most similar report that I have access to is the following: \n Unique ID: {retrieved_documents[0][0].metadata['uid']} \n Content: {retrieved_documents[0][0].page_content}")

if st.session_state.button_clicked == "Retrieve Report and Image":
    retrieved_documents, retrieved_images = retrieve_similar_report(vector_db_text, '/home/ccig/Desktop/Nuno/rag_data/query_test.json', text_collection, vector_db_images, image_collection)
    img_path = plot_images(retrieved_images)
    st.write(f"The most similar report that I have access to is the following:")
    st.write(f"Unique ID: {retrieved_documents[0][0].metadata['uid']}")
    st.write(f"Content: {retrieved_documents[0][0].page_content}")
    st.write(f"Found {len(retrieved_images['ids'])} images related to the report")
    st.write(f"Images shown below:")
    st.image(img_path, caption="Retrieved Image", use_column_width="auto")

if st.session_state.button_clicked == "Delete DB":
    st.write("Feature still not implemented")

if st.session_state.button_clicked == "Delete Report from DB":
    st.write("Feature still not implemented")

# React to user input
if message := st.chat_input("Ask me anything"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": message})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(message)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        retriever = get_relevant_documents(vector_db_text, message, 0.2, text_collection)
        print(f"Documents with distance below the threshold: {len(retriever)}")
        

        runnable = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

        # Get the documents relevant to the message, extract text, and update the context
        used_context = [doc[0] for doc in retriever]
        
        #used_pdfs = list(set([doc.metadata.get('source') for doc in used_context]))

        #writer.update_context(text_from_documents)

        #print(f"Number of retrieved documents: {len(used_context)}")
        
        #n=len(used_context)
        
        #await cl.Message(content=f" Number of documents found with a cosine distance below `{threshold}`: `{similars}` \n Used documents from: {used_pdfs}").send()
        #print( f"The cosine distance for each of the `{n}` documents: \n `{[score for doc_id, score in search_scores[:n]]}")
                    
        ai_response_content =[]
        # Run the runnable pipeline asynchronously. This will generate a response from the language model for each question in the message content.
        for chunk in runnable.stream(
            {"question": message},
            config=RunnableConfig(),
        ): 
        
            ai_response_content.append(chunk)
            response = st.write_stream(ai_response_content)
        
        #full_response = "".join(ai_response_content)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
