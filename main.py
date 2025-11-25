import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
import spacy
import csv
import os


# -------------------- SETUP --------------------
GEMINI_API_KEY = "AIzaSyCpVuZQksbWbdW7fQ_LoVDSndRENmDj_fY"
genai.configure(api_key=GEMINI_API_KEY)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

nlp = spacy.load("en_core_web_sm")

# -------------------- UTILS --------------------
def call_gemini(prompt: str):
    """Safe, clean Gemini call wrapper."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        if resp and resp.text:
            return resp.text.strip()
        return ""
    except Exception as e:
        return f"ERROR: {e}"

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return " ".join(tokens)

def extract_text_from_file(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + " "
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    return text

# -------------------- LLM: ICP KEYWORDS --------------------
def extract_keywords_llm(text):
    prompt = f"""
You are an ICP Keyword Extractor.

Extract ONLY commercially valuable ICP keywords from the text:

Include:
- industry sectors
- buyer roles & titles
- functional pain points
- capabilities
- technical stack

Rules:
- NO verbs
- NO sentences
- ONLY nouns and noun phrases
- NO filler words
- NO duplicates
- Return ONLY a comma-separated list

TEXT:
{text}
"""
    return call_gemini(prompt)

# -------------------- LLM: SUMMARY --------------------
def summarize_text(text):
    prompt = f"""
You are a Senior Market Research Analyst.

Summarize the company text into exactly 5 bullet points:

1. Core Offering
2. Ideal Buyers
3. Business Value
4. Key Differentiator
5. Revenue Model

Return ONLY the 5 bullet points.

TEXT:
{text}
"""
    return call_gemini(prompt)

# -------------------- NER --------------------
def extract_ner(text):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE"]]
    return list(set(ents))

# -------------------- CLEANING --------------------
def clean_keywords(keywords):
    stop = set([
        "data", "research", "product", "industry", "company",
        "service", "technology", "platform", "solution", 
        "system", "software", "application", "team"
    ])

    cleaned = []
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        if len(kw) < 3:
            continue
        if kw in stop:
            continue
        if not re.match("^[a-zA-Z\s]+$", kw):
            continue
        cleaned.append(kw)

    return sorted(set(cleaned))

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="LeadGen Intelligence Engine", layout="wide")

if "final_keywords" not in st.session_state:
    st.session_state.final_keywords = []

if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = set()

st.title("📌 LeadGen Intelligence Engine")

with st.container(border=True):
    location = st.text_input("Location")
    job_title = st.text_input("Job Title")
    industry = st.text_input("Industry")
    about = st.text_area("About (Company Description)", height=160)
    uploads = st.file_uploader("Upload Files", accept_multiple_files=True)
    run = st.button("Run Analysis", type="primary", use_container_width=True)

summary = ""

# -------------------- PROCESS --------------------
if run:
    all_text = about

    if uploads:
        for f in uploads:
            all_text += " " + extract_text_from_file(f)

    # LLM keyword extraction
    llm_raw = extract_keywords_llm(all_text)
    llm_keywords = [k.strip() for k in llm_raw.split(",") if k.strip()]

    # NER
    ner = extract_ner(all_text)

    # Summary
    summary = summarize_text(all_text)
    st.session_state.summary = summary

    # MERGING (LLM dominates)
    merged = set(llm_keywords)
    merged.update([k.lower() for k in ner])

    st.session_state.final_keywords = clean_keywords(list(merged))

    st.success("✅ Analysis Complete!")

# -------------------- DISPLAY --------------------
st.subheader("📌 Summary")
st.write(st.session_state.get("summary", ""))

st.subheader("📌 Manage Keywords")

cols = st.columns(3)

for i, kw in enumerate(st.session_state.final_keywords):
    col = cols[i % 3]
    with col:
        checked = st.checkbox(kw, key=f"kw_{kw}", value=kw in st.session_state.selected_keywords)
        if checked:
            st.session_state.selected_keywords.add(kw)
        else:
            st.session_state.selected_keywords.discard(kw)

# DELETE BUTTON
if st.button("❌ Delete Selected Keywords", type="secondary"):
    st.session_state.final_keywords = [
        k for k in st.session_state.final_keywords
        if k not in st.session_state.selected_keywords
    ]
    st.session_state.selected_keywords = set()
    st.rerun()

# FINAL OUTPUT
st.subheader("📌 Final Keywords")
st.write(st.session_state.final_keywords)


def save_keywords_to_csv(keywords, filename="keywords.csv"):
    os.makedirs("data", exist_ok=True)   # Save all CSVs inside /data folder
    filepath = os.path.join("data", filename)

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Keyword"])
        for kw in keywords:
            writer.writerow([kw])

    print(f"Saved {len(keywords)} keywords to {filepath}")
    

def save_summary_to_csv(summary_text, filename="summary.csv"):
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)

    # Split summary into bullet points
    lines = [line.strip() for line in summary_text.split("\n") if line.strip()]

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Summary Point"])
        for line in lines:
            writer.writerow([line])

    print(f"Saved {len(summary_text)} summary topics to {filepath}")


def save_initial_input_to_csv(input_dict, filename="input_data.csv"):

    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)

    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(["Field", "Value"])   # Header

        for key, value in input_dict.items():
            writer.writerow([key, value])

    print(f"Saved input data to {filepath}")


if __name__ == "__main__":

    # Example keywords list
    keywords_list = st.session_state.final_keywords

    # Example summary topics
    summary_list = st.session_state.get("summary", "")

    # Example initial user input
    user_input = {
        "Location": location,
        "Job Title": job_title,
        "Industry": industry,
        "About": about,
        "Uploaded Files": ", ".join([f.name for f in uploads]) if uploads else ""
    }

    # Save all CSVs
    save_keywords_to_csv(keywords_list)
    save_summary_to_csv(summary_list)
    save_initial_input_to_csv(user_input)
