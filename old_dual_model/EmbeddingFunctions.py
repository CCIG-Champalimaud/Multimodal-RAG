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
import math
import matplotlib.pyplot as plt
import tempfile

def initialize_multimodal_vector_database(path, persist_directory, collection_names=['documents', 'images']):
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
                      collection_name=collection_names[0],
                      collection_metadata={"hnsw:space": "cosine"})
    vectordb_images = Chroma(persist_directory=persist_directory+'_images',
                      client=client,
                      embedding_function=embedding_function,
                      collection_name=collection_names[1],
                      collection_metadata={"hnsw:space": "cosine"})

    # Get the initial count of documents in the collection and set a retriever
    print(f"Number of text documents in the database: {text_collection.count()}")
    print(f"Number of image documents in the database: {image_collection.count()}")
    return vectordb_text, vectordb_images, text_collection, image_collection

def add_NLMCXR_to_vectorstore(dataset_file, text_collection, image_collection):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
    
    #'~/Desktop/Nuno/rag_data/NLMCXR_png'
    # Load the text from the uploaded PDF
    
    embedder = ReportAndImageEmbeddings()
    
    
    #dataset = open(dataset_file.path)
    dataset = open(dataset_file)
    dataset = json.load(dataset)
    
    for entry in dataset:
        # join report fields into  a single set of texts
        text = ''.join([dataset[entry]['report'][description] for description in dataset[entry]['report'].keys()])
        
        embedding_text = embedder.embed_reports([text])
        text_collection.add(
                ids=[entry],
                embeddings=[embedding_text],
                documents=[text],
                metadatas=[{"source": "NLMCXR", "uid": entry, 
                            'n_images': len(list(dataset[entry]['images'].keys())) if 'images' in dataset[entry] else 0}]
        )

        if 'images' in dataset[entry]:
            img_embeddings = []
            info = {
                'source': [],
                'uid': [],
                'uid_img': [],
                'caption': [],
                'path': []
            }
            for i, img in enumerate(dataset[entry]['images'].keys()):
                info['source'].append('NLMCXR')
                info['uid'].append(entry)
                info['uid_img'].append(img)
                info['caption'].append(dataset[entry]['images'][img]['caption'])
                info['path'].append(dataset[entry]['images'][img]['path'])
                image = process_image(dataset[entry]['images'][img]['path'])
                embedding_img = embedder.embed_images(image)
                img_embeddings.append(embedding_img)
            
        
            image_collection.add(
                ids=[entry+'_'+str(img_num) for img_num in range(len(list(dataset[entry]['images'].keys())))],
                embeddings=img_embeddings,
                documents=list(dataset[entry]['images'].keys()),
                #metadatas=[{"source": "NLMCXR", "uid":entry ,"uid_img": imgid} for imgid in imgs_ids]
                metadatas=flatten_dict(info)
            )

            
# to flaten a multi dict into a list of single dicts, for metadata
def flatten_dict(input_dict):
    # Get the number of entries for each key
    n = len(next(iter(input_dict.values())))
    
    # Create the list of dictionaries
    return [
        {key: input_dict[key][i] for key in input_dict}
        for i in range(n)
    ]


def retrieve_similar_report(vector_db, query, collection, vector_db_images=None, image_collection=None):

    query_data = open(query)
    query_data = json.load(query_data)

    key = list(query_data.keys())[0]

    text = ''.join([query_data[key]['report'][description] for description in query_data[key]['report'].keys()])

    search_results = vector_db.similarity_search_with_score(query=[text])
    
    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    print(search_results)

    if image_collection == None:
        return search_results
    else:
        img_ids = [ search_results[0][0].metadata['uid']+'_'+str(i) for i in range(search_results[0][0].metadata['n_images'])]
        retireved = image_collection.get(ids=img_ids)

        return search_results, retireved

def get_relevant_documents(vector_db, query, threshold, collection):
    search_results = vector_db.similarity_search_with_score(query, collection.count())
    search_results = sorted(search_results, key=lambda x: x[1], reverse=True)

    return search_results


def plot_images(images):
    num_images = len(images['ids'])
    cols = math.ceil(math.sqrt(num_images))  # Number of columns in the grid
    rows = (num_images // cols) + (num_images % cols > 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten()
    
    for i in tqdm(range(num_images), desc="Processing images"):
        axes[i].imshow(process_image(images['metadatas'][i]['path']), cmap='gray')
        axes[i].set_title(images['metadatas'][i]['caption'])
        axes[i].axis('off')

    # Hide any empty subplots
    for j in range(i + 1, rows * cols):
        axes[j].axis('off')
    
    #plt.suptitle(f'{title}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle

    tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    full_temp_path = os.path.abspath(tmp_file.name)
    
    plt.savefig(full_temp_path)  # Save the plot as an image file

    tmp_file.close()

    return full_temp_path


def process_image(image_file):
    image_file = os.path.expanduser(image_file)
    print(f"\nProcessing {image_file}\n")
    image_data = Image.open(image_file).convert('RGB')
            
    return image_data


class ReportAndImageEmbeddings():
    def __init__(self):
        #hf_AaHkBxdNvbEdtmmocTdqMyrhckxHVzuAiZQZ
        self.vision_model = AutoModel.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(torch.device("cuda:2")).eval()
        self.vision_processor = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5', trust_remote_code=True)
        
        self.text_model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)
        
        # Check if the model has the expected method
        #if not hasattr(self.vision_model, 'get_image_features'):
        #    raise AttributeError("The vision model does not have a 'get_image_features' method. Please choose a compatible model.")

    def embed_images(self, image):
        pixel_values = self.vision_processor(images=image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).to(torch.device("cuda:2"))

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

def get_number_relevant_documents_old(vector_db, query, threshold, collection):
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