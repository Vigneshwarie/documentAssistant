# documentAssistant

## Description
documentAssistant is an application designed to chunk documents, store them in the Pinecone vector database, and enable conversational interaction using OpenAI's language model. This allows users to chat with their documents effortlessly.

## Features
**Document Chunking:** Splits documents into manageable chunks for efficient storage and retrieval.   
**Vector Storage:** Stores document embeddings in the Pinecone vector database for quick and accurate search.   
**Conversational Interaction:** Enables seamless querying and interaction with documents using OpenAI's language model.   

## Prerequisites
Before running the application, ensure you have the following:
- Python 3.7+   
- Pinecone API key   
- OpenAI API key   
- Required Python packages (listed in requirements.txt)   

## Installation
### 1. Clone the repository
```
git clone https://github.com/Vigneshwarie/documentAssistant.git
cd documentassistant
```

### 2. Create and activate a virtual environment
```
python3 -m venv venv
source venv/bin/activate  
```

### 3. Install the required packages
```
pip install -r requirements.txt
```

## Configuration
### Set up environment variables
Create a .env file in the root directory of the project and add your Pinecone and OpenAI API keys:
```
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Run the Application
Start the application:
```
streamlit run app.py
```

## Credits
- [OpenAI](https://www.openai.com/) for providing the language model.   
- [Pinecone](https://www.pinecone.io) for the vector database.   
- [Hugging Face](https://huggingface.co) for the SentenceTransformer model.   

## License
 ![Github license](https://img.shields.io/badge/license-MIT-blue.svg) 