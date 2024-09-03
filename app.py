import streamlit as st
# from PyPDF2 import PdfReader
import spacy
from spacy.cli import download as spacy_download
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_transformers import EmbeddingsRedundantFilter
# from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import os
from dotenv import load_dotenv
import re
import fitz  # PyMuPDF
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import numpy as np
# from transformers import BlipProcessor, BlipForConditionalGeneration
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
# from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Ensure SpaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load the API key
load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CLIP model initialization
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize BLIP model and processor
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Gemini model initialization
gemini_model = GeminiMultiModal(model_name="models/gemini-1.5-flash")

# Updated AdvancedPDFReader class
class AdvancedPDFReader:
    def __init__(self, pdf_files):
        self.pdf_files = pdf_files
        self.text = ""
        self.tables = []
        self.images = []

    def extract_text(self, document):
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            self.text += page.get_text("text")
            self.tables += self.extract_tables(page)

    def extract_tables(self, page):
        tables = []
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[4] == 1:  # Check if the block is part of a table
                tables.append(block[4])
        return tables

    def extract_images(self, document, images_dir='extracted_images/'):
        # Ensure the directory for images exists
        os.makedirs(images_dir, exist_ok=True)

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = os.path.join(images_dir, f'image_{page_num+1}_{img_index+1}.{image_ext}')
                with open(image_name, "wb") as image_file:
                    image_file.write(image_bytes)
                self.images.append(image_name)  # Append the image path to the images list

    def read_pdf(self):
        for file in self.pdf_files:
            document = fitz.open(stream=file.read(), filetype="pdf")
            self.extract_text(document)
            self.extract_images(document)  # Extract images after text
            document.close()
        return self.text, self.tables, self.images

# Clean and preprocess text
def clean_text(text):
    text = " ".join(text.split())
    text = re.sub(r'\[\d+\]', '', text)
    return text

# Preprocess and segment text into sentences
def preprocess_text(text):
    text = clean_text(text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

# Chunk text into segments of approximately 512 tokens with 50-token overlap
def chunk_text(sentences, max_tokens=512, overlap=50):
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        tokens = len(sentence.split())
        if current_length + tokens > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = sum(len(sent.split()) for sent in current_chunk)
        
        current_chunk.append(sentence)
        current_length += tokens
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Document chunking using the preprocessing and chunking strategy
def get_chunks(text):
    sentences = preprocess_text(text)
    chunks = chunk_text(sentences)
    return chunks

# Create Complete Object for MultiQuery Retrieval
class LineList(BaseModel):
    lines: List[str] = Field(description="Lines of text")

def create_complete_object(raw_lines: str, question: str) -> str:
    lines = [line.strip() for line in raw_lines.strip().split('\n') if line.strip()]
    line_list = LineList(lines=lines)
    
    return f"""
You are an advanced Artificial Intelligence assistant skilled in extracting and analyzing information from various types of documents, including text, tables, and all forms of images such as charts, graphs, diagrams, flowcharts, and examples. These images are considered part of the document content and may contain important data relevant to the query.

### Response Strategy:

1. **Consistent and Comprehensive Review**: For queries related to the document, thoroughly review the entire document, including text, tables, and all images. If you identify relevant data in images such as charts or graphs, explicitly mention and reference these figures (e.g., "Figure 4"). Do not contradict yourself by claiming the absence of data that you later reference.

2. **Avoid Contradictions**: Ensure that your response is consistent throughout. If a relevant chart or graph is present in an image, acknowledge its presence from the outset and include the data from both the table and the graph where applicable. Do not state that something is absent if you will later reference it.

3. **Explicit Figure and Table Referencing**: Clearly identify and reference specific figures and tables when they contain relevant data. For instance, if a query relates to a graph, mention both the figure number and the data extracted from it. Similarly, ensure tables are accurately referenced and their data included in the response.

4. **Balance Detail and Relevance**: For document-specific queries, aim for detailed, clear, and accurate answers that include all relevant data from text, tables, and images. For general questions, provide concise and relevant information using your internal knowledge as needed.

5. **Clarify and Explain**: Make sure all explanations are clear and well-structured. When referencing images or data from the document, include relevant details and provide clear context to avoid any ambiguity. Ensure that responses are informative and appropriate for the context of the question.

### Objective: Deliver accurate, consistent, and comprehensive answers based on the document when applicable, and address general queries with relevant and concise information. Ensure that all forms of images, including charts, graphs, diagrams, and other visuals, are considered and correctly referenced without contradiction.

Information:
{line_list}

Question:
{question}??? Analyze the document thoroughly, including all forms of images, to provide a detailed and consistent response. If relevant data is found in images such as charts or graphs, reference them clearly and avoid contradicting earlier statements. For general questions, offer a relevant and informative answer based on your internal knowledge and the document content if it enhances the response. AND never mention the document in response.

Answer:
    """

# Post-Processing to Refine AI Responses
def refine_response(response_text):
    # Additional processing to ensure clarity, correctness, and completeness
    refined_text = response_text.strip()
    if refined_text[-1] != '.':
        refined_text += '.'
    return refined_text

# Streamlit app logic
def main():
    st.title("Document and Image Retrieval App")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        pdf_reader = AdvancedPDFReader(uploaded_files)
        text, tables, image_paths = pdf_reader.read_pdf()
        
        st.write("Processing completed. You can now search for specific information.")
        print(image_paths)
        # Display extracted images
        for image_path in image_paths:
            st.image(image_path)

        st.write("Enter your query to retrieve information:")

    user_query = st.text_input("Enter your query:")
    if user_query:
        complete_object = create_complete_object(text, user_query)
        # load image documents from local directory
        image_documents = SimpleDirectoryReader('extracted_images/').load_data()
        # Use Gemini model to complete the query based on the context
        response = gemini_model.complete(
            prompt=complete_object,
            image_documents=image_documents  # Pass image paths directly
        )
        
        # refined_response = refine_response(response.get("text", ""))
        st.write(response.text)

if __name__ == "__main__":
    main()
