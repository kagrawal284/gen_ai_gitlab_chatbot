# Gen AI Chatbot – Local Setup Guide

This is a **Streamlit-based Retrieval-Augmented Generation (RAG) chatbot** that answers user questions based on the contents of provided web links. It uses **LangChain**, **Gemini API**, and a **vectorstore** (e.g., FAISS) to retrieve relevant chunks and generate accurate answers.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kagrawal284/gen_ai_gitlab_chatbot
cd gen_ai_gitlab_chatbot
```

### 2. Set up virtual environment(optional but recommended)

```bash
conda create -n venv_chatbot python=3.10
conda activate venv_chatbot
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a .env file and add the following

```bash
GOOGLE_API_KEY=your_google_api_key

USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
```

> ### Steps to Create `GOOGLE_API_KEY`
>
> - Go to [https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key)
> - Click on **Get API key** and then **Create API Key**

### 5. Run the app locally

```bash
streamlit run main.py
```

### 6. Public url of app

```bash
https://bot-genai.streamlit.app/
```
