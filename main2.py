import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import os

# -------------------- SETUP GEMINI API --------------------
GEMINI_API_KEY = "AIzaSyD9ASQUjcFBTf80Tqusq_xxNgFNefmuTmM"

genai.configure(api_key=GEMINI_API_KEY)

# Download required NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- HELPER FUNCTIONS --------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + " "
    elif uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    return text

# -------------------- KEYWORD EXTRACTION (Gemini) --------------------
def extract_keywords_gemini(clean_text):
    prompt = f"""
    ### ROLE & GOAL
    You are an expert Text Analyst and B2B Marketing Strategist. Your primary goal is to analyze the provided text (which may be a company description, product document, or 'About' page) and extract a list of high-intent, specific, and descriptive keywords and keyphrases.

    These keywords will be used for lead generation. They must capture the company's core identity, what it sells, who it sells to, and the problems it solves.

    ### KEYWORD CATEGORIES TO EXTRACT
    Analyze the text and extract keywords that fall into these specific categories:

    1.  **Core Products & Services:** What do they actually sell?
        * (e.g., "AI-driven platform," "SaaS solution," "data analytics tool," "CRM software")

    2.  **Key Features & Capabilities:** What does their product *do*?
        * (e.g., "demand forecasting," "inventory management," "automated billing," "workflow automation")

    3.  **Target Industry & Domain:** What specific industries do they serve?
        * (e.g., "healthcare," "fintech," "e-commerce," "supply chain logistics," "real estate")

    4.  **Target Audience & Customer Profile:** *Who* are their customers?
        * (e.g., "B2B clients," "enterprise," "small businesses," "software engineers," "marketing managers")

    5.  **Technologies & Jargon:** What specific technologies, standards, or acronyms do they use or mention?
        * (e.g., "machine learning," "AWS," "Azure," "API," "HIPAA-compliant," "ERP," "ISO 27001")

    6.  **Problems Solved & Value Proposition:** What *problems* do they solve or what *value* do they provide? (Focus on the *concept*, not the action).
        * (e.g., "cost reduction," "workflow optimization," "revenue growth," "risk management," "data security")

    ### STRICT OUTPUT FORMATTING

    * **ONLY** return a single, comma-separated list.
    * **DO NOT** include any preamble, explanation, or titles (e.g., do not write "Here is the list:").
    * **DO NOT** use bullet points or newlines.
    * **NEW, CRITICAL RULE: DO NOT return full sentences, clauses, or any text containing verbs** (e.g., 'we help you optimize' is WRONG. 'workflow optimization' is CORRECT). Focus exclusively on nouns, noun phrases, and technical acronyms.
    * **DO NOT** include generic, non-descriptive, or common "stopwords" (e.g., 'we', 'our', 'the', 'is', 'a', 'for', 'company', 'solution', 'technology'). Only return the specific terms.
    * **DO** combine words into meaningful noun phrases (e.g., "supply chain optimization" is better than "supply," "chain," "optimization").

    ---

    ### HIGH-QUALITY EXAMPLE

    **Input Text:**
    "Our advanced, AI-driven platform helps e-commerce businesses optimize their supply chain. We use proprietary machine learning models for demand forecasting and inventory management, which reduces costs for our B2B clients. Our solution is fully scalable on AWS and is GDPR-compliant."

    **Your Output (This is the format you MUST follow):**
    AI-driven platform, e-commerce, supply chain optimization, proprietary machine learning models, demand forecasting, inventory management, cost reduction, B2B clients, scalable, AWS, GDPR-compliant

    ---

    ### TEXT TO ANALYZE:
    {clean_text}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"⚠️ Gemini generation error: {e}")
        return ""

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="User Information Form", page_icon="📋", layout="wide")

if 'keywords_list' not in st.session_state:
    st.session_state.keywords_list = []

st.title("User Information Form")
st.markdown("Please fill out the details below to submit your information.")
st.markdown("---")

with st.container(border=True):

    location = st.text_input("Location", placeholder="e.g., San Francisco, CA or Remote")
    job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    company_size = st.selectbox(
        "Company Size",
        ["", "1-10 employees", "11-50 employees", "51-200 employees", "201-500 employees",
         "501-1000 employees", "1001-5000 employees", "5001-10,000 employees", "10,001+ employees"]
    )
    industry = st.text_input("Industry", placeholder="e.g., Technology, Finance, Healthcare")
    About = st.text_area("About", placeholder="Provide your company description here...", height=200)
    uploaded_docs = st.file_uploader(
        "Upload Docs", type=None, accept_multiple_files=True,
        help="Upload any relevant documents."
    )

    st.markdown("---")
    st.subheader("Keywords")

    def add_keyword_callback():
        new_keyword = st.session_state.keyword_input_box.strip()
        if new_keyword and new_keyword not in st.session_state.keywords_list:
            st.session_state.keywords_list.append(new_keyword)
        st.session_state.keyword_input_box = ""

    st.text_input(
        "Keywords", placeholder="Type a keyword and press Enter...",
        key='keyword_input_box', on_change=add_keyword_callback,
        help="Type a keyword and press Enter to add it as a tag below."
    )

    st.session_state.keywords_list = st.multiselect(
        "Your Keywords",
        options=st.session_state.keywords_list,
        default=st.session_state.keywords_list,
        help="This box shows your added keywords. Click the 'x' on any tag to remove it."
    )

    st.markdown("---")
    submit_button = st.button("Submit Information", use_container_width=True, type="primary")

# -------------------- SUBMISSION LOGIC --------------------
if submit_button:
    st.success("Form submitted successfully!")
    st.balloons()

    full_text = About
    if uploaded_docs:
        for f in uploaded_docs:
            full_text += " " + extract_text_from_file(f)

    cleaned_text = preprocess_text(full_text)
    keywords = extract_keywords_gemini(cleaned_text)

    submitted_data = {
        "Location": location,
        "Job Title": job_title,
        "Company Size": company_size,
        "Industry": industry,
    }

    Keywords_content = {
        "About": About,
        "Cleaned Text": cleaned_text,
        "Keywords": keywords,
        "Uploaded Files": [file.name for file in uploaded_docs] if uploaded_docs else []
    }

    st.subheader("Extracted Keywords")
    st.write(Keywords_content)

    # Delete old data files before saving new data
    for file_name in ["keywords_data.csv", "saved_data.csv"]:
        if os.path.exists(file_name):
            os.remove(file_name)

    pd.DataFrame([submitted_data]).to_csv("saved_data.csv", index=False)
    pd.DataFrame([keywords]).to_csv("keywords_data.csv", index=False)

    st.success("✅ Data saved to keywords_data.csv (appended)")
