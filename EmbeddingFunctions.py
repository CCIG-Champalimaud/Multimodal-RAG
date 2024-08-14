from langchain_community.document_loaders import PyPDFLoader
import chromadb
import ollama  # Import the missing module
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
from tqdm import tqdm
import json
from PIL import Image
import os
from io import BytesIO
from langchain.embeddings.base import Embeddings
from transformers import AutoProcessor, AutoModel, AutoTokenizer, AutoFeatureExtractor, CLIPImageProcessor
import torch
import torch.nn.functional as F
import numpy as np

def initialize_multimodal_vector_database(path, collection_name, persist_directory,framework, model):
    # Create a persistent client and a collection for the vector database
    client = chromadb.PersistentClient(path=path)
    text_collection = client.get_or_create_collection(
        name='documents',
        metadata={"hnsw:space": "cosine"})
    image_collection = client.get_or_create_collection(
        name='images',
        metadata={"hnsw:space": "cosine"})

    embedding_function = ReportAndImageEmbeddings()

    # Load the previously stored documents into a Chroma vectorstore
    vectordb_text = Chroma(persist_directory=persist_directory+'_text',
                      client=client,
                      embedding_function=embedding_function,
                      collection_name='documents',
                      collection_metadata={"hnsw:space": "cosine"})
    vectordb_images = Chroma(persist_directory=persist_directory+'_images',
                      client=client,
                      embedding_function=embedding_function,
                      collection_name='images',
                      collection_metadata={"hnsw:space": "cosine"})

    # Get the initial count of documents in the collection and set a retriever
    print(f"Number of text documents in the database: {text_collection.count()}")
    print(f"Number of image documents in the database: {image_collection.count()}")
    return vectordb_text, vectordb_images, text_collection, image_collection

def add_NLMCXR_to_vectorstore(dataset_file, text_collection, image_collection,  DELIMITER,
                                   embeddings_framework, embeddings_model_text, embeddings_model_img):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
    
    #'~/Desktop/Nuno/rag_data/NLMCXR_png'
    # Load the text from the uploaded PDF
    
    embedder = ReportAndImageEmbeddings()
    
    
    #dataset = open(dataset_file.path)
    dataset = open(dataset_file)
    dataset = json.load(dataset)

    print(text_collection.count())
    print(image_collection.count())
    
    for entry in dataset:
        # join report fields into  a single set of texts
        text = ''.join([dataset[entry]['report'][description] for description in dataset[entry]['report'].keys()])
        
        embedding_text = embedder.embed_reports([text])
        text_collection.add(
                ids=[entry],
                embeddings=[embedding_text],
                documents=[text]
        )

        print(text_collection.count())
        print(image_collection.count())

        if 'images' in dataset[entry]:
            img_embeddings = []
            for img in dataset[entry]['images'].keys():
                img_idx = dataset[entry]['images'][img]
                image = process_image(dataset[entry]['images'][img]['path'])
                embedding_img = embedder.embed_images(image)
                img_embeddings.append(embedding_img)
        
            image_collection.add(
                ids=[entry+'_'+img_name for img_name in dataset[entry]['images'].keys()],
                embeddings=img_embeddings,
                documents=list(dataset[entry]['images'].keys())
            )



def retrieve_similar_report(vector_db, query, threshold, collection):

    query_data = open(query)
    query_data = json.load(query_data)

    key = list(query_data.keys())[0]

    text = ''.join([query_data[key]['report'][description] for description in query_data[key]['report'].keys()])

    print('General DB check:', vector_db.get())
    print('Text collection DB check:', vector_db.get("documents"))
    print('Image collection DB check:', vector_db.get("images"))

    print('query text:', text)
    print('collection:', collection)
    print('collection count:', collection.count())
    search_results = vector_db.similarity_search_with_score(query=[text])
    print('search results:', search_results)
    '''
    # Extract the scores from the search results
    relevant_count = 0
    for doc_id, score in search_results:
        #print(score)
        if score < threshold:
            relevant_count += 1
    # Return the count of relevant documents
    return relevant_count
    '''
    asdsa

    

def process_image(image_file):
    image_file = os.path.expanduser(image_file)
    print(f"\nProcessing {image_file}\n")
    image_data = Image.open(image_file).convert('RGB')
            
    return image_data


class ReportAndImageEmbeddings():
    def __init__(self):
        #hf_AaHkBxdNvbEdtmmocTdqMyrhckxHVzuAiZQZ
        self.vision_model = AutoModel.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).cuda().eval()
        self.vision_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5', trust_remote_code=True)
        
        self.text_model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)
        
        # Check if the model has the expected method
        #if not hasattr(self.vision_model, 'get_image_features'):
        #    raise AttributeError("The vision model does not have a 'get_image_features' method. Please choose a compatible model.")

    def embed_images(self, image):
        pixel_values = self.vision_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        output = self.vision_model(pixel_values)
        #print(output.last_hidden_state.shape)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.element_size() * output.last_hidden_state.nelement())
        output = output.pooler_output.cpu().detach()
        return output[-1].tolist()
            
    def embed_reports(self, text):
        text_embeddings = self.text_model.encode(text, instruction="", max_length=4096)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
        #print(text)
        #print(text_embeddings.shape)
        #print(text_embeddings.element_size() * text_embeddings.nelement())
        return text_embeddings[-1].tolist()

    def embed_query(self, document):
        #print(type(document))
        if isinstance(document[0], str):
            return self.embed_reports(document)
        elif isinstance(document[0], list):
            return self.embed_images(document)
        else:
            raise ValueError("Task must be either 'text' or 'image'")
    

def add_pdf_to_vectorstore_complete(text_file, id_str, collection, DELIMITER, embeddings_framework, embeddings_model, chunk_size, chunk_overlap):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
    loader = PyPDFLoader(text_file.path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    texts = [str(text) for text in texts]

    # Extracting file name from text_file
    file_name = text_file.name

    for i, d in tqdm(enumerate(texts)):
        if embeddings_framework == "ollama":
            response = ollama.embeddings(model=embeddings_model, prompt=d)
            embedding = response["embedding"]
        elif embeddings_framework == "huggingface":
            model = HuggingFaceEmbeddings(model_name=embeddings_model)
            embedding = model.embed_query(d)
        else:
            print("this embedding framework is not implemented yet")
        
        id = id_str + DELIMITER + str(i)
        #print(id)

        # Adding file name as metadata
        metadata = {"source": file_name}

        collection.add(
            ids=[id],
            embeddings=[embedding],
            documents=[d],
            metadatas=[metadata]  # Add metadata here
        )

def read_config(config_path: str) -> dict[str, str]:
    with open(config_path) as o:
        return yaml.safe_load(o)
    

def get_embeddings_model(framework: str, model: str):
    if framework == 'huggingface':
        return HuggingFaceEmbeddings(model_name=model)
    elif framework == 'ollama':
        return OllamaEmbeddings(model=model)
    else:
        raise NotImplementedError("...")


def get_chat_model(framework: str, model: str, temperature: int = None):
    if framework == 'huggingface':
        return HuggingFaceEmbeddings(model_name=model)
    elif framework == "ollama":
        return Ollama(model=model, temperature=0)
    else:
        raise NotImplementedError("...")
    
def get_document_prefixes(collection, DELIMITER):
        # Get the documents in the collection
        documents = collection.get()
        ids = documents['ids']

        # Store id prefixes in a list
        pref_list = []
        for id in ids:
            prefix = id.split(DELIMITER)[0]
            if prefix not in pref_list:
                pref_list.append(prefix)
        return pref_list

def get_number_relevant_documents(vector_db, query, threshold, collection):
    search_results = vector_db.similarity_search_with_score(query, collection.count())
    #print(search_results)
    # Extract the scores from the search results
    relevant_count = 0
    for doc_id, score in search_results:
        #print(score)
        if score < threshold:
            relevant_count += 1
    # Return the count of relevant documents
    return relevant_count