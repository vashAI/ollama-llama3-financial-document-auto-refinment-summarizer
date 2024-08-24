import streamlit as st
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

# Initialize the LLaMA model via Ollama
model_id = "llama3.1"
model = Ollama(model=model_id)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def generate_refined_summary(model, text, max_iterations=5):
    iteration = 0
    current_summary = None
    questions_generated = True

    # Step 1: Generate an initial summary
    initial_prompt_text = """
    You are a summarization expert. Your task is to create summaries.
    Here is the original document text:
    
    {text}
    
    Start by creating a summary in your first pass. 
    
    Create an initial summary.
    """
    initial_prompt = ChatPromptTemplate.from_template(initial_prompt_text)
    current_summary = model(initial_prompt.format(text=text))

    initial_summary = current_summary

    # Iterative refinement process
    while iteration < max_iterations and questions_generated:
        iteration += 1
        # Step 2: Ask the LLM to compare the original text and summary, and to refine it
        refinement_prompt_text = """
        You are a summarization expert. Your task is to refine summaries.
        Here is the original document text:
        
        {text}
        
        Refine the below current summary, keep it as it is but ensure it becomes more complete, coherent, clear, and accurate. 
        Aim to capture the essence of the text with each refinement.
        
        Current summary:
        {summary}
        
        Please provide a refined summary below:
        """
        refinement_prompt = ChatPromptTemplate.from_template(refinement_prompt_text)
        current_summary = model(refinement_prompt.format(text=text, summary=current_summary))

    return initial_summary, current_summary

# Streamlit interface
st.title('Document Summarization Refinement')

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("Document loaded successfully!")

    iterations = st.slider("Select the number of iterations for refinement", min_value=1, max_value=10, value=5)
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summaries..."):
            initial_summary, final_summary = generate_refined_summary(model, document_text, max_iterations=iterations)
            
            # Displaying initial and final summary side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Initial Summary")
                st.markdown(initial_summary)

            with col2:
                st.subheader(f"Final Summary after {iterations} iterations")
                st.markdown(final_summary)
