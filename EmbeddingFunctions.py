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

    # isto tem de desaparecer depois, e mesmo esta parte da imagem tem de ser modificada, maybe por um path nos metadados
    dset_path = '/home/ccig/Desktop/Nuno/rag_data/NLMCXR_pdf'

    image_index = [retrieved[i]["page_num"] - 1 for i in range(len(retrieved))]

    images = [convert_from_path(os.path.join(dset_path, f'{metadata[retrieved[i]["doc_id"]]["file_name"]}.pdf')) for i in range(len(retrieved))]


    content = [{"type": "image", "image": images[i][image_index[i]]} for i in range(len(retrieved))]
    content.append({"type": "text", "text": query})

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
        
    print('OK1',torch.cuda.memory_summary())

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print('OK1',torch.cuda.memory_summary())

    image_inputs, video_inputs = process_vision_info(messages)

    print('OK1',torch.cuda.memory_summary())

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    print('OK1',torch.cuda.memory_summary())

    inputs = inputs.to("cuda:3")
        
    print('OK1',torch.cuda.memory_summary())

    generated_ids = llm.generate(**inputs, max_new_tokens=200)

    print(torch.cuda.memory_summary())

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text