import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Function to initialize various models
def initialize_models():
    text_generator = pipeline("text-generation", model="gpt2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")  # Updated model
    sentiment_analyzer = pipeline("sentiment-analysis")
    question_answerer = pipeline("question-answering")
    return text_generator, summarizer, sentiment_analyzer, question_answerer

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function to initialize Chroma client and create a collection
def initialize_chroma():
    try:
        client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
        collection = client.create_collection("pdf_documents")
        return client, collection
    except RuntimeError as e:
        st.error(f"Chroma initialization failed: {e}")
        st.error("Please ensure that SQLite is updated to version 3.35 or higher.")
        return None, None  # Return None if initialization fails

# Function to add documents to the vector database
def add_to_vector_db(collection, pdf_text):
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    embedding = embedding_fn.embed_documents([pdf_text])
    collection.add(documents=[pdf_text], embeddings=embedding)

# Function to query the vector database
def query_vector_db(collection, query):
    results = collection.query(query=query, n_results=5)
    return results

# Function to summarize text
def summarize_text(summarizer, text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to analyze sentiment
def analyze_sentiment(sentiment_analyzer, text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]

# Function to answer questions based on context
def answer_question(question_answerer, context, question):
    result = question_answerer(question=question, context=context)
    return result['answer']

# Function to display responses
def display_response(response):
    st.info(response)

# Main function to run the Streamlit app
def main():
    st.title('Advanced PDF Chat Application with Vector Database and NLP Tasks')

    # Initialize models and Chroma DB
    text_generator, summarizer, sentiment_analyzer, question_answerer = initialize_models()
    client, collection = initialize_chroma()

    if collection is None:  # If Chroma initialization failed, exit early
        return

    # File uploader for PDF files
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # Extract text from the uploaded PDF
        pdf_text = extract_text_from_pdf(pdf_file)
        st.success("PDF text extracted successfully!")

        # Add extracted text to vector database
        add_to_vector_db(collection, pdf_text)
        st.success("PDF content added to vector database!")

        # User input for queries based on extracted text or vector database search
        user_input = st.text_input("Ask a question about the PDF content:")

        if st.button('Generate Response'):
            if user_input:
                # Querying vector database for relevant documents based on user input
                db_results = query_vector_db(collection, user_input)
                if db_results:
                    context_input = f"{db_results[0]} \n\nUser Question: {user_input}"
                    response = generate_response(text_generator, context_input)
                    display_response(response)
                else:
                    st.warning("No relevant documents found in the database.")
            else:
                st.warning("Please enter a question.")

        # Summarization option
        if st.button('Summarize PDF Content'):
            summary = summarize_text(summarizer, pdf_text)
            display_response(summary)

        # Sentiment Analysis option
        if st.button('Analyze Sentiment of User Input'):
            sentiment_result = analyze_sentiment(sentiment_analyzer, user_input)
            display_response(f"Sentiment: {sentiment_result['label']} (Score: {sentiment_result['score']:.2f})")

        # Question Answering option
        if st.button('Answer Specific Question'):
            question = st.text_input("Enter your specific question:")
            if question:
                answer = answer_question(question_answerer, pdf_text, question)
                display_response(answer)
            else:
                st.warning("Please enter a specific question.")

if __name__ == "__main__":
    main()

