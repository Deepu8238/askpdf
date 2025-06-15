import streamlit as st
import fitz  # PyMuPDF
import openai
from openai import OpenAI
import os
import re
import json
import time
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="StudyBuddyAI",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state with secrets
if "current_card_index" not in st.session_state:
    st.session_state.current_card_index = 0
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "memory_index" not in st.session_state:
    st.session_state.memory_index = None
if "concept_embeddings" not in st.session_state:
    st.session_state.concept_embeddings = []
if "concept_texts" not in st.session_state:
    st.session_state.concept_texts = []
if "generated_mcqs" not in st.session_state:
    st.session_state.generated_mcqs = []
if "mcq_answers" not in st.session_state:
    st.session_state.mcq_answers = {}
if "theme" not in st.session_state:
    st.session_state.theme = st.secrets.ui.default_theme
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = st.secrets.app.similarity_threshold
if "max_concepts" not in st.session_state:
    st.session_state.max_concepts = st.secrets.app.max_concepts
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# Initialize the sentence transformer model
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    try:
        if st.session_state.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                model = SentenceTransformer('all-MiniLM-L6-v2')
                st.session_state.embedding_model = model
        return st.session_state.embedding_model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

def initialize_memory():
    """Initialize or load the FAISS index for concept memory"""
    try:
        if st.session_state.memory_index is None:
            # Create a new FAISS index
            dimension = 384  # dimension of the embedding
            st.session_state.memory_index = faiss.IndexFlatL2(dimension)
            # Load existing concepts if available
            load_concepts()
    except Exception as e:
        st.error(f"Error initializing memory: {str(e)}")

def load_concepts():
    """Load saved concepts from disk"""
    try:
        if Path('concepts.pkl').exists():
            with open('concepts.pkl', 'rb') as f:
                data = pickle.load(f)
                st.session_state.concept_embeddings = data['embeddings']
                st.session_state.concept_texts = data['texts']
                if len(st.session_state.concept_embeddings) > 0:
                    st.session_state.memory_index.add(np.array(st.session_state.concept_embeddings))
    except Exception as e:
        st.error(f"Error loading concepts: {str(e)}")

def save_concepts():
    """Save concepts to disk"""
    try:
        data = {
            'embeddings': st.session_state.concept_embeddings,
            'texts': st.session_state.concept_texts
        }
        with open('concepts.pkl', 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"Error saving concepts: {str(e)}")

def add_concept_to_memory(text, embedding_model):
    """Add a new concept to memory"""
    try:
        # Get embedding for the new concept
        embedding = embedding_model.encode([text])[0]
        
        # Check similarity with existing concepts
        if len(st.session_state.concept_embeddings) > 0:
            similarities = faiss.IndexFlatIP(384).search(
                np.array([embedding]), 
                np.array(st.session_state.concept_embeddings), 
                1
            )[0][0]
            
            if similarities > 0.8:  # Threshold for considering concepts similar
                return False, "Concept already exists in memory"
        
        # Add to memory
        st.session_state.concept_embeddings.append(embedding)
        st.session_state.concept_texts.append(text)
        st.session_state.memory_index.add(np.array([embedding]))
        save_concepts()
        return True, "New concept added to memory"
    except Exception as e:
        return False, f"Error adding concept: {str(e)}"

def analyze_concepts(text, embedding_model):
    """Analyze text for new concepts and compare with existing ones"""
    try:
        # Split text into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
        
        new_concepts = []
        existing_concepts = []
        
        for sentence in sentences:
            success, message = add_concept_to_memory(sentence, embedding_model)
            if success:
                new_concepts.append(sentence)
            else:
                existing_concepts.append(sentence)
        
        return new_concepts, existing_concepts
    except Exception as e:
        st.error(f"Error analyzing concepts: {str(e)}")
        return [], []

# Get OpenAI client
def get_openai_client():
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        if not api_key or api_key == "your-api-key-here":
            st.error("Please set your OpenAI API key in .streamlit/secrets.toml")
            return None
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

# Extract text from PDF
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file):
    try:
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Summarize full text using GPT-3.5
@st.cache_data(show_spinner=False)
def summarize_full_text(_client, text):
    try:
        response = _client.chat.completions.create(
            model=st.secrets.app.default_model,
            messages=[
                {"role": "system", "content": "You are a concise educational summarizer."},
                {"role": "user", "content": f"Summarize this content in 3-4 paragraphs:\n{text}"},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error summarizing text: {str(e)}")
        return ""

# Generate quiz + flashcards from full summary using GPT-4
def generate_quiz_flashcards(_client, full_summary, num_questions):
    try:
        prompt = f"""
        Based on this combined course material summary:

        {full_summary}

        Generate:
        - A 3-sentence refined summary
        - {num_questions} MCQs (each with 4 options, mark the correct one with an asterisk *)
        - 5 flashcards (Format: Front: ... | Back: ...)

        Format:
        ## Summary
        ...

        ## Multiple Choice Questions
        1. ...
        a) Option A
        b) Option B
        c) Option C
        d) Option D*

        ## Flashcards
        Front: ... | Back: ...
        """
        response = _client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return ""

# Parse flashcards
def parse_flashcards(text):
    try:
        cards = re.findall(r"Front:\s*(.*?)\s*\|\s*Back:\s*(.*?)(?=\nFront:|$)", text, re.DOTALL)
        return [{"question": q.strip(), "answer": a.strip()} for q, a in cards]
    except Exception as e:
        st.error(f"Error parsing flashcards: {str(e)}")
        return []

# Parse MCQs
def parse_mcqs(text):
    try:
        mcqs = []
        # Updated pattern to better match MCQ format
        pattern = r"(\d+)\.\s*(.*?)\n\s*a\)\s*(.*?)\n\s*b\)\s*(.*?)\n\s*c\)\s*(.*?)\n\s*d\)\s*(.*?)(?:\*)?(?:\n|$)"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            num, question, a, b, c, d = match.groups()
            # Clean up the options
            options = [opt.strip() for opt in [a, b, c, d]]
            # Find the correct answer (marked with *)
            correct_option = None
            for i, opt in enumerate([a, b, c, d]):
                if "*" in opt:
                    correct_option = i
                    options[i] = opt.replace("*", "").strip()
            
            if correct_option is not None:
                mcqs.append({
                    "number": int(num),
                    "question": question.strip(),
                    "options": options,
                    "correct_index": correct_option
                })
        
        return mcqs
    except Exception as e:
        st.error(f"Error parsing MCQs: {str(e)}")
        return []

# Main app
def main():
    st.title("🤖 StudyBuddyAI")
    st.markdown("Upload your PDF and get a full summary, MCQs, and flashcards — all in one go!")

    # Initialize memory system
    initialize_memory()
    embedding_model = get_embedding_model()

    client = get_openai_client()

    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        # Theme Settings
        st.write("**Theme Settings**")
        theme = st.selectbox(
            "Select Theme",
            ["light", "dark"],
            index=0 if st.session_state.theme == "light" else 1,
            key="theme_selector"
        )
        
        # Apply theme change
        if theme != st.session_state.theme:
            st.session_state.theme = theme
            # Force a rerun to apply the theme
            st.rerun()
        
        # Model Settings
        st.write("**Model Settings**")
        model_choice = st.selectbox(
            "Model for summarization",
            ["gpt-3.5-turbo", "gpt-4"],
            index=0 if st.secrets.app.default_model == "gpt-3.5-turbo" else 1,
            key="model_choice"
        )
        num_questions = st.slider(
            "# of MCQs",
            min_value=3,
            max_value=10,
            value=st.secrets.app.default_num_questions,
            step=1,
            key="num_questions"
        )
        
        # Memory Settings
        st.write("**Memory Settings**")
        similarity_threshold = st.slider(
            "Concept Similarity Threshold",
            min_value=0.5,
            max_value=0.95,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="Higher values mean concepts need to be more similar to be considered duplicates"
        )
        if similarity_threshold != st.session_state.similarity_threshold:
            st.session_state.similarity_threshold = similarity_threshold
        
        max_concepts = st.number_input(
            "Maximum Concepts to Store",
            min_value=100,
            max_value=10000,
            value=st.session_state.max_concepts,
            step=100,
            help="Maximum number of concepts to store in memory"
        )
        if max_concepts != st.session_state.max_concepts:
            st.session_state.max_concepts = max_concepts
        
        # Memory Management
        if st.secrets.memory.persist_memory:
            st.write("**Memory Management**")
            if st.button("Clear Memory", help="Clear all stored concepts"):
                st.session_state.concept_embeddings = []
                st.session_state.concept_texts = []
                st.session_state.memory_index = None
                initialize_memory()
                st.success("Memory cleared successfully!")
            
            # Memory Stats
            if st.secrets.ui.show_memory_stats:
                st.subheader("📊 Memory Stats")
                st.write(f"Concepts in memory: {len(st.session_state.concept_texts)}")
                if st.session_state.concept_texts:
                    st.write(f"Memory usage: {len(st.session_state.concept_texts) / st.session_state.max_concepts * 100:.1f}%")
            
            # Export/Import Memory
            st.write("**Memory Backup**")
            if st.button("Export Memory", help="Export memory to a file"):
                save_concepts()
                st.success("Memory exported successfully!")
            
            uploaded_memory = st.file_uploader("Import Memory", type="pkl", help="Import previously saved memory")
            if uploaded_memory:
                try:
                    data = pickle.load(uploaded_memory)
                    st.session_state.concept_embeddings = data['embeddings']
                    st.session_state.concept_texts = data['texts']
                    st.session_state.memory_index = None
                    initialize_memory()
                    st.success("Memory imported successfully!")
                except Exception as e:
                    st.error(f"Error importing memory: {str(e)}")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and client:
        # Only generate new content if we don't have it already
        if not st.session_state.generated_mcqs:
            with st.spinner("Extracting and processing text..."):
                text = extract_text_from_pdf(uploaded_file)
                
                # Analyze concepts
                new_concepts, existing_concepts = analyze_concepts(text, embedding_model)
                
                full_summary = summarize_full_text(client, text)
                result = generate_quiz_flashcards(client, full_summary, num_questions)
                flashcards = parse_flashcards(result)
                mcqs = parse_mcqs(result)
                
                # Store the generated content
                st.session_state.generated_mcqs = mcqs
                st.session_state.flashcards = flashcards
                st.session_state.result = result
                st.session_state.new_concepts = new_concepts
                st.session_state.existing_concepts = existing_concepts
                st.session_state.summary = full_summary
        else:
            # Use stored content
            mcqs = st.session_state.generated_mcqs
            flashcards = st.session_state.flashcards
            result = st.session_state.result
            new_concepts = st.session_state.new_concepts
            existing_concepts = st.session_state.existing_concepts
            full_summary = st.session_state.summary

        st.success("✅ Content Generated!")

        # Display concept analysis
        if new_concepts or existing_concepts:
            st.subheader("🧠 Concept Analysis")
            if new_concepts:
                st.write("**New Concepts:**")
                for concept in new_concepts:
                    st.info(concept)
            if existing_concepts:
                st.write("**Previously Learned Concepts:**")
                for concept in existing_concepts:
                    st.success(concept)

        # Display Summary
        st.subheader("📝 Summary")
        summary_match = re.search(r"## Summary\n(.*?)\n##", result, re.DOTALL)
        if summary_match:
            st.markdown(summary_match.group(1).strip())

        # Interactive MCQs
        st.subheader("📚 Multiple Choice Questions")
        if mcqs:
            for mcq in mcqs:
                with st.expander(f"Question {mcq['number']}"):
                    st.write(f"**{mcq['question']}**")
                    
                    # Initialize answer for this question if not exists
                    if mcq['number'] not in st.session_state.mcq_answers:
                        st.session_state.mcq_answers[mcq['number']] = None
                    
                    # Create unique key for each question
                    key = f"mcq_{mcq['number']}"
                    
                    # Use the stored answer or create new radio button
                    selected_option = st.radio(
                        "Select your answer:",
                        mcq['options'],
                        key=key,
                        index=None if st.session_state.mcq_answers[mcq['number']] is None 
                        else mcq['options'].index(st.session_state.mcq_answers[mcq['number']])
                    )
                    
                    # Update stored answer when user makes a selection
                    if selected_option:
                        st.session_state.mcq_answers[mcq['number']] = selected_option
                        
                        if mcq['options'].index(selected_option) == mcq['correct_index']:
                            st.success("✅ Correct!")
                        else:
                            st.error(f"❌ Incorrect. The correct answer is: {mcq['options'][mcq['correct_index']]}")
        else:
            st.warning("No MCQs were generated. Please try again with different content.")

        # Add a reset button
        if st.button("Reset Quiz"):
            st.session_state.mcq_answers = {}
            st.rerun()

        # Interactive Flashcards
        if flashcards:
            st.subheader("🧠 Interactive Flashcards")
            card = flashcards[st.session_state.current_card_index]

            st.write(f"Card {st.session_state.current_card_index + 1} of {len(flashcards)}")

            if not st.session_state.show_answer:
                st.info(card["question"])
                if st.button("Show Answer"):
                    st.session_state.show_answer = True
                    st.rerun()
            else:
                st.success(card["answer"])
                if st.button("Show Question"):
                    st.session_state.show_answer = False
                    st.rerun()

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous"):
                    st.session_state.current_card_index = (st.session_state.current_card_index - 1) % len(flashcards)
                    st.session_state.show_answer = False
                    st.rerun()
            with col2:
                if st.button("Next ➡️"):
                    st.session_state.current_card_index = (st.session_state.current_card_index + 1) % len(flashcards)
                    st.session_state.show_answer = False
                    st.rerun()

            # Download options
            st.download_button(
                "📥 Download Flashcards (JSON)",
                json.dumps(flashcards, indent=2),
                file_name="flashcards.json"
            )

        st.download_button(
            "📄 Download Full Output",
            result,
            file_name="ai_tutor_output.txt"
        )

if __name__ == "__main__":
    main()
