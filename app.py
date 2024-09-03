# import streamlit as st
# from PyPDF2 import PdfReader
# import spacy
# from spacy.cli import download as spacy_download
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.chains import LLMChain
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.vectorstores import Chroma
# from langchain.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
# from langchain.retrievers.document_compressors import EmbeddingsFilter, DocumentCompressorPipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.storage import InMemoryStore
# from langchain.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import List, Union, Dict
# import os
# from dotenv import load_dotenv
# import re
# import fitz  # PyMuPDF
# import torch
# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import numpy as np
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from llama_index.multi_modal_llms.gemini import GeminiMultiModal

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# # Ensure SpaCy model is downloaded
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     spacy_download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")

# # Load the API key
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # CERT model initialization
# cert_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# cert_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Updated AdvancedPDFReader class
# class AdvancedPDFReader:
#     def __init__(self, pdf_files):
#         self.pdf_files = pdf_files
#         self.text = ""
#         self.tables = []
#         self.images = []

#     def extract_text(self, document):
#         for page_num in range(len(document)):
#             page = document.load_page(page_num)
#             self.text += page.get_text("text")
#             self.tables += self.extract_tables(page)

#     def extract_tables(self, page):
#         tables = []
#         blocks = page.get_text("blocks")
#         for block in blocks:
#             if block[4] == 1:  # Check if the block is part of a table
#                 tables.append(block[4])
#         return tables

#     def extract_images(self, document, images_dir='extracted_images/'):
#         # Ensure the directory for images exists
#         os.makedirs(images_dir, exist_ok=True)

#         for page_num in range(len(document)):
#             page = document.load_page(page_num)
#             image_list = page.get_images(full=True)
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = document.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]
#                 image_name = os.path.join(images_dir, f'image_{page_num+1}_{img_index+1}.{image_ext}')
#                 with open(image_name, "wb") as image_file:
#                     image_file.write(image_bytes)
#                 self.images.append(image_name)  # Append the image path to the images list

#     def read_pdf(self):
#         for file in self.pdf_files:
#             document = fitz.open(stream=file.read(), filetype="pdf")
#             self.extract_text(document)
#             self.extract_images(document)  # Extract images after text
#             document.close()
#         return self.text, self.tables, self.images
    
#     def get_image_embeddings(self):
#         image_embeddings = []
#         for image_path in self.images:
#             image = Image.open(image_path).convert("RGB")
#             inputs = cert_processor(images=image, return_tensors="pt")
#             with torch.no_grad():
#                 outputs = cert_model.get_image_features(**inputs)
#             image_embeddings.append(outputs.squeeze().numpy())
#         return image_embeddings


# # Clean and preprocess text
# def clean_text(text):
#     text = " ".join(text.split())
#     text = re.sub(r'\[\d+\]', '', text)
#     return text

# # Preprocess and segment text into sentences
# def preprocess_text(text):
#     text = clean_text(text)
#     doc = nlp(text)
#     sentences = [sent.text for sent in doc.sents]
#     return sentences

# # Chunk text into segments of approximately 512 tokens with 50-token overlap
# def chunk_text(sentences, max_tokens=512, overlap=50):
#     chunks = []
#     current_chunk = []
#     current_length = 0
    
#     for sentence in sentences:
#         tokens = len(sentence.split())
#         if current_length + tokens > max_tokens:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = current_chunk[-overlap:]
#             current_length = sum(len(sent.split()) for sent in current_chunk)
        
#         current_chunk.append(sentence)
#         current_length += tokens
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     return chunks

# # Document chunking using the preprocessing and chunking strategy
# def get_chunks(text):
#     sentences = preprocess_text(text)
#     chunks = chunk_text(sentences)
#     return chunks

# def adjust_embedding_dimension(embedding, target_dim):
#     if len(embedding) < target_dim:
#         # Zero-pad the embedding
#         padding = np.zeros(target_dim - len(embedding))
#         return np.concatenate([embedding, padding])
#     elif len(embedding) > target_dim:
#         # Truncate the embedding (less common)
#         return embedding[:target_dim]
#     return embedding

# # Create Embeddings Store
# def get_vector_store(text_chunks, table_chunks, image_embeddings, pdf_reader):
#     if not text_chunks and not table_chunks and not image_embeddings:
#         raise ValueError("No data to embed. Please check the input text and images.")
    
#     # Create embeddings for text and tables
#     text_table_chunks = text_chunks + table_chunks
#     text_table_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     text_table_embedding_vectors = text_table_embeddings.embed_documents(text_table_chunks)
    
#     # Create embeddings for images
#     image_embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     image_embedding_vectors = image_embeddings_model.embed_documents(["image"])  # Dummy input to get embeddings

#     # Adjust dimensions
#     text_table_dim = len(text_table_embedding_vectors[0])
#     image_dim = len(image_embedding_vectors[0])
#     target_dim = max(text_table_dim, image_dim)

#     adjusted_text_table_embeddings = [adjust_embedding_dimension(emb, target_dim) for emb in text_table_embedding_vectors]
#     adjusted_image_embeddings = [adjust_embedding_dimension(emb, target_dim) for emb in image_embedding_vectors]

#     # Create vector stores
#     text_table_store = FAISS.from_embeddings(list(zip(text_table_chunks, adjusted_text_table_embeddings)), text_table_embeddings)
#     image_store = FAISS.from_embeddings(list(zip(pdf_reader.images, adjusted_image_embeddings)), image_embeddings_model)

#     # Save vector stores
#     text_table_store.save_local("text_table_faiss_index")
#     image_store.save_local("image_faiss_index")



# # Document chunking using the preprocessing and chunking strategy
# def get_conversation_chain_pdf():
#     prompt_template = """
# You are an advanced Artificial Intelligence assistant designed to provide accurate and resource-efficient answers using both textual and visual information. Your goal is to deliver precise responses by leveraging all available context, including both text and images.

# ### Response Strategy:

# 1. **Utilize Both Text and Image Contexts**: When answering questions, use both the textual information and image descriptions provided. Ensure that responses incorporate relevant details from both types of data.

# 2. **Prioritize Accuracy and Relevance**: Focus on delivering responses that are accurate and directly relevant to the question. If the question pertains to images, integrate information from image descriptions along with text.

# 3. **Resource-Conscious Answering**: Keep responses concise and to the point, minimizing unnecessary processing while ensuring all relevant details are included.

# 4. **Detailed Image Information**: If the question relates to specific images, refer to the provided image contexts and descriptions. Make sure to include or address information about the images as needed.

# ### Objective: Provide accurate, concise answers using both text and image contexts. Ensure that responses are relevant and resource-efficient, integrating information from both types of data when applicable.

# Information:
# {context}

# Question:
# {question}??? Provide a precise, accurate answer using both the text and image information available. Ensure that responses are well-informed and integrate details from both sources as necessary.

# Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.55)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


# # Create Complete Object for MultiQuery Retrieval
# class LineList(BaseModel):
#     lines: List[str] = Field(description="Lines of text")

# def create_complete_object(raw_lines: str, question: str) -> Dict[str, Union[str, LineList]]:
#     lines = [line.strip() for line in raw_lines.strip().split('\n') if line.strip()]
#     line_list = LineList(lines=lines)
#     return {'question': question, 'text': line_list}

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an advanced scientific AI language model assistant with a focus on generating diverse and comprehensive queries in passive voice in a formal and scientific way to retrieve information just as if a machine is asking questions in desire of a detailed explanation and extract precise info from context, general knowledge, scientific reasoning, and external information. Your task is to create eight to twelve distinct versions of the user's question to maximize the chances of finding relevant documents or generating a well-informed answer.

# For each version, consider different angles, possible variations, and broader or narrower interpretations of the question. Make sure they are in passive voice and formal and scientific in tone, just like an AI asking questions in desire of a detailed explanation. Include versions that might elicit context-specific responses, general knowledge, scientific reasoning, and even external information if the context is insufficient.

# Ensure that at least one variant is optimized to retrieve scientific answers, particularly if the question pertains to a well-known scientific concept, even if the context is not specific or relevant.

# Provide these alternative questions separated by newlines.

# Original question: {question}???
#     """
# )

# # Post-Processing to Refine AI Responses
# def refine_response(response_text):
#     # Additional processing to ensure clarity, correctness, and completeness
#     refined_text = response_text.strip()
#     if refined_text[-1] != '.':
#         refined_text += '.'
#     return refined_text

# # Initialize BLIP model and processor
# blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def get_image_captions(image_paths):
#     captions = []
#     for image_path in image_paths:
#         image = Image.open(image_path).convert("RGB")
#         inputs = blip_processor(images=image, return_tensors="pt")
#         out = blip_model.generate(**inputs)
#         caption = blip_processor.decode(out[0], skip_special_tokens=True)
#         captions.append(caption)
#     return captions

# # Processing User Input
# def user_input(user_query, retrieval_method, pdf_reader, embeddings):
#     # Load vector stores for text/tables and images
#     text_table_vector_store = FAISS.load_local("faiss_text_table_index", embeddings, allow_dangerous_deserialization=True)
#     image_vector_store = FAISS.load_local("faiss_image_index", embeddings, allow_dangerous_deserialization=True)

#     # Initialize GeminiMultiModal
#     gemini_multimodal = GeminiMultiModal(
#         text_vector_store=text_table_vector_store,
#         image_vector_store=image_vector_store,
#         llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.45)
#     )

#     # Handling greetings and general questions directly
#     if user_query.lower() in ["hello", "hi"]:
#         st.write("AI_Response:", "Hello! How can I assist you today?")
#         return

#     # Use GeminiMultiModal for multimodal retrieval
#     response = gemini_multimodal.query(user_query)

#     # Display the response
#     st.write("AI_Response:")
#     st.markdown(response["text_response"], unsafe_allow_html=True)

#     # Display the relevant image if one is identified in the response
#     if response["image_response"]:
#         st.image(response["image_response"], caption=f"Relevant Image", use_column_width=True)


# def main():
#     # Load PDF and create embeddings
#     pdf_reader = load_pdf("path_to_pdf")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
#     # Process the PDF and images
#     text_chunks, table_chunks, image_embeddings = process_pdf_and_images(pdf_reader)
    
#     # Create vector stores
#     get_vector_store(text_chunks, table_chunks, image_embeddings, pdf_reader, embeddings)
    
#     # User input handling
#     user_query = st.text_input("Enter your query:")
#     if user_query:
#         user_input(user_query, retrieval_method, pdf_reader, embeddings)


# if __name__ == "__main__":
#     main()
import streamlit as st
# from PyPDF2 import PdfReader
import spacy
from spacy.cli import download as spacy_download
import google.generativeai as genai
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
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
# import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CLIP model initialization
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize BLIP model and processor
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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



