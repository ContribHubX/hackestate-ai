import os
import json
from dotenv import load_dotenv
from typing import List, Tuple, Dict
from huggingface_hub import login
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
load_dotenv()

def retrieve_listings() -> List[any] :
    print("running list retrieval")
    with open("datasets/filipinohomes_listing.json", "r", encoding='utf-8') as f:
        listings = json.load(f)
        return listings;



def flatten_dict(d: Dict[str, any], parent_key: str = '', sep: str = '.') -> Dict[str, any]:
    """
    Recursively flattens a nested dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def prepare_text_and_metadata(listings: List[dict]) -> Tuple[List[str], List[dict]]:
    print("Preparing texts and metadata")

    texts: List[str] = []
    metadatas: List[dict] = []

    for listing in listings:
        # Flatten the entire listing dict (with nested keys like "details.bedrooms", etc.)
        flat_listing = flatten_dict(listing)

        # Create the text from flattened key-value pairs
        text = "\n".join(f"{key.replace('.', ' ').title()}: {value}" for key, value in flat_listing.items())
        texts.append(text.strip())

        # Store flattened metadata
        metadatas.append(flat_listing)

    return texts, metadatas


def create_and_persist_to_db(texts: List[str], metadatas: List[dict], persist_directory: str) -> None:
    print("creating and persisting database")
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db: VectorStore = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_directory
    )
    db.persist()



def load_db(persist_directory: str) -> VectorStore:
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


def load_llm_model(model_name: str, token: str) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
  print("loading llm model")
  login(token)

  tokenizer: PreTrainedTokenizer  = AutoTokenizer.from_pretrained(model_name, token=token)
  tokenizer.padding_side = "left"
  tokenizer.pad_token = tokenizer.eos_token

  model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",
      torch_dtype="auto",
      low_cpu_mem_usage=True,
      token=token
  )

  return (tokenizer, model)




def run_embeddings() -> None:
    listings = retrieve_listings()
    texts, metadatas =  prepare_text_and_metadata(listings)
    create_and_persist_to_db(texts, metadatas, os.getenv("VECTOR_DB"))




