from langchain_community.document_loaders import PyPDFLoader

import ollama  # Import the missing module
from langchain_community.embeddings import OllamaEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama
from tqdm import tqdm
import json
from PIL import Image
import os
from glob import glob
from io import BytesIO
from langchain.embeddings.base import Embeddings
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import tempfile
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.utils.colpali_processing_utils import process_images, process_queries
from colpali_engine.utils.image_from_page_utils import load_from_dataset
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
from byaldi import RAGMultiModalModel

def initialize_multimodal_vector_database():
    # Create a persistent client and a collection for the vector database
    if os.path.exists(".byaldi/Documents"):
        RAG = RAGMultiModalModel.from_index("Documents", verbose=0)
        metadata = json.load(open("metadata.json"))
        metadata = {int(key): value for key, value in metadata.items()}

        return RAG, metadata
    else:
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali", device='cuda:2', verbose=0)

    return RAG, {}

def add_NLMCXR_to_vectorstore(RAG):
    # Add the text from the uploaded PDF to the vectorstore by embedding it and adding it to the collection
    
    files = glob(os.path.join('/home/ccig/Desktop/Nuno/rag_data/NLMCXR_pdf_micro', '*.pdf'))

    uids = list(range(len(files)))

    report_ids = [file.split('/')[-1].split('.pdf')[0] for file in files]

    metadata = {uids[i]: {'file_name':report_ids[i]} for i in range(len(uids))}

    for i, file in enumerate(files):

        RAG.index(
            input_path=file,
            index_name='Documents', # index will be saved at index_root/index_name/
            doc_ids=[uids[i]],
            store_collection_with_index=True,
            overwrite=True,
            #metadata=metadata,

        )

    json_data = json.dumps(metadata, indent = 4)
    with open("metadata.json", "w") as outfile:
        outfile.write(json_data)

    return metadata


def create_generator_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",trust_remote_code=True, torch_dtype=torch.bfloat16).to('cuda:3').eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

    return model, processor
     

def retrieve_similar_report(RAG, query, k=1):

    results = RAG.search(query, k=k)

    return results

def generate_reply(llm, processor, query, retrieved, metadata):

    image_index = retrieved[0]["page_num"] - 1

    image = convert_from_path(os.path.join('/home/ccig/Desktop/Nuno/rag_data/NLMCXR_pdf', f'{metadata[retrieved[0]["doc_id"]]["file_name"]}.pdf'))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image[image_index],
                },
                {"type": "text", "text": query},
            ],
        }
    ]
        

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda:3")
        

    generated_ids = llm.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

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


class MedicalReportsEmbedding():
    def __init__(self):
        self.model_name = "vidore/colpali-v1.2"
        self.model = ColPali.from_pretrained("vidore/colpaligemma-3b-pt-448-base", torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.model.load_adapter(self.model_name)
        self.model = self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def embed_documents(self, image):

        processed_image = process_images(self.processor, image)

        #print(processed_image)

        processed_image = {k: v.to(self.model.device) for k, v in processed_image.items()}

        with torch.no_grad():
            embeddings_doc = self.model(**processed_image)

        print(embeddings_doc.shape)
        return torch.unbind(embeddings_doc.to("cpu"))[0].tolist()

        #pixel_values = self.vision_processor(images=image, return_tensors='pt').pixel_values
        #pixel_values = pixel_values.to(torch.bfloat16).to(torch.device("cuda:2"))

        #output = self.vision_model(pixel_values)
        #print(output.last_hidden_state.shape)
        #print(output.pooler_output.shape)
        #print(output.last_hidden_state.element_size() * output.last_hidden_state.nelement())
        #output = output.pooler_output.cpu().detach()
        #return output[-1].tolist()

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