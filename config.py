"""
Configuration for the Legal RAG Bench evaluation.

- Loads API keys from `.env`.
- Creates LangChain embedding + chat model clients.
- Exposes lists of embedding/generative models for evaluation runs.
"""

import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_isaacus import IsaacusEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Base folder used elsewhere to store and load vector databases.
DB_DIR = "databases" 

# Load `.env` into environment variables (no-op if already set in the shell).
load_dotenv()

# Helper function to load environment variables.
def _require_env(var_name: str) -> str:
    """
    Read a required environment variable.

    Raises:
        ValueError: if the variable is missing or empty.
    """
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not set in .env")
    return value

# --------------------
# API keys (required)
# --------------------
ISAACUS_API_KEY = _require_env("ISAACUS_API_KEY")
OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")

# --------------------
# Embedding models
# To evaluate more embedding models, initialize embedding models via LangChain,
# Then add the model + dict to `embedding_models` list.
# --------------------

# OpenAI text-embedding-3-large
te3l = {
    "model": OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=OPENAI_API_KEY,
    ),
    "model_name": "te3l",
}

# Isaacus Kanon 2 Embedder
kanon2 = {
    "model": IsaacusEmbeddings(
        model="kanon-2-embedder",
        api_key=ISAACUS_API_KEY,
    ),
    "model_name": "kanon2",
}

# Google Gemini embedding 001
gemini001 = {
    "model": GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY,
    ),
    "model_name": "gemini001",
}

# 
embedding_models = [te3l, kanon2, gemini001]

# --------------------
# Generative models
# To evaluate more generative models, initialize generative models via LangChain,
# then add the model + dict to `generative_models` list.
# --------------------

# GPT 5.2
gpt52 = {
    "model": ChatOpenAI(
        model="gpt-5.2",
        api_key=OPENAI_API_KEY,
    ),
    "model_name": "gpt52",
}

# GPT 3
gemini31 = {
    "model": ChatGoogleGenerativeAI(
        model="gemini-3.1-pro-preview",
        google_api_key=GOOGLE_API_KEY,
    ),
    "model_name": "gemini31pro",
}

generative_models = [gpt52, gemini31]

# --------------------
# Judge model
# The eval code only supports one judge model, however, you are free to change,
# this model by initializing any LangChain chat model client here.
# --------------------

# GPT 5.2 high reasoning
judge_model = {
    "model": ChatOpenAI(
        model="gpt-5.2",
        api_key=OPENAI_API_KEY,
        reasoning_effort="high",  
        seed=42,                
    ),
    "model_name": "gpt52_judge",
}

# --------------------
# Retrieval at k=?
# The eval code will iterate through Ks=[...] for each k,
# This is so you can easily compare performance across k.
# --------------------
Ks = [5]