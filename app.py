import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI
import os
import re
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="AI Tutor & Quiz Generator", layout="wide")

# Session state init
if "show_answer" not in st.session_state:
    st.session_state.show_answer = False
if "current_card_index" not in st.session_state:
    st.session_state.current_card_index = 0

# Get OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found in environment variables.")
        return None
    try:
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
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a concise educational summarizer."},
                {"role": "user", "content": f"Summarize this content in 3-4 paragraphs:\n{text}"},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error summarizing full text: {e}"

# Generate quiz + flashcards from full summary using GPT-4
def generate_quiz_flashcards(client, full_summary, num_questions):
    try:
        prompt = f"""
        Based on this combined course material summary:

        {full_summary}

        Generate:
        - A 3-sentence refined summary
        - {num_questions} MCQs (each with 4 options, mark the correct one)
        - 5 flashcards (Format: Front: ... | Back: ...)

        Format:
        ## Summary
        ...

        ## Multiple Choice Questions
        1. ...
        2. ...

        ## Flashcards
        Front: ... | Back: ...
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating quiz: {e}"

# Parse flashcards
def parse_flashcards(text):
    cards = re.findall(r"Front:\s*(.*?)\s*\|\s*Back:\s*(.*?)\n", text, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in cards]

# Main app
def main():
    st.title("🤖 AI Tutor: Full PDF Quiz & Flashcard Generator")
    st.markdown("Upload your PDF and get a full summary, MCQs, and flashcards — all in one go!")

    client = get_openai_client()

    with st.sidebar:
        st.subheader("Settings")
        model_choice = st.selectbox("Model for summarization", ["gpt-3.5-turbo", "gpt-4"])
        num_questions = st.slider("# of MCQs", 3, 10, 5)

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and client:
        with st.spinner("Extracting and processing text..."):
            text = extract_text_from_pdf(uploaded_file)
            full_summary = summarize_full_text(client, text)
            result = generate_quiz_flashcards(client, full_summary, num_questions)
            flashcards = parse_flashcards(result)

        st.success("✅ Content Generated!")

        # Display content
        st.subheader("🎉 AI Output")

        # Extract and show only the MCQs section
        mcq_section = re.search(r"## Multiple Choice Questions\n(.*?)\n## Flashcards", result, re.DOTALL)
        if mcq_section:
            st.markdown("### 📝 Multiple Choice Questions")
            st.markdown(mcq_section.group(1))

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

            st.download_button("📥 Download Flashcards (JSON)", json.dumps(flashcards, indent=2), file_name="flashcards.json")

        st.download_button("📄 Download Full Output", result, file_name="ai_tutor_output.txt")

if __name__ == "__main__":
    main()
