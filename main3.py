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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# -------------------- SETUP GEMINI API --------------------
GEMINI_API_KEY = "AIzaSyD9ASQUjcFBTf80Tqusq_xxNgFNefmuTmM"
genai.configure(api_key=GEMINI_API_KEY)

# NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# -------------------- PREPROCESSING --------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return " ".join(tokens)

# -------------------- FILE READING --------------------
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

# -------------------- LLM KEYWORD EXTRACTION --------------------
def extract_keywords_llm(clean_text):
    prompt = f"""
    ### ROLE & GOAL
    You are a Senior B2B Sales Operations Strategist and Data Enrichment Specialist. Your goal is to analyze company documentation to generate high-precision targeting keywords for lead generation campaigns.

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

    
    ### OBJECTIVE
    Analyze the text above and extract a comma-separated list of keywords and phrases that describe the **Ideal Customer Profile (ICP)** and the **Market Segment**. 
    
    Do not just extract words present in the text; infer the industry terms, technical standards, and business categories that a sales professional would use to find companies that fit this profile.

    ### EXTRACTION RULES (CRITICAL)
    1. **Nouns & Phrases Only:** No verbs, no full sentences. (e.g., Use "Supply Chain Automation" NOT "We automate supply chains").
    2. **No Fluff:** Remove marketing buzzwords like "best," "cutting-edge," "leading," "world-class," "state-of-the-art."
    3. **Specific Categories:** Ensure keywords cover these four dimensions:
       - **Industry/Verticals:** (e.g., FinTech, Commercial Real Estate, Horeca)
       - **Buyer Personas:** (e.g., Chief Risk Officer, VP of Engineering, Fleet Manager)
       - **Tech Stack/Integrations:** (e.g., Salesforce, Kubernetes, Shopify, SAP)
       - **Pain Points/Solutions:** (e.g., Churn Reduction, GDPR Compliance, Inventory Shrinkage)
    4. **Deduplicate:** Do not repeat similar terms.
    5. **Formatting:** Return ONLY the raw comma-separated string. No bullet points, no introduction, no "Here are the keywords."

    ### EXAMPLE OUTPUT
    SaaS, B2B Marketing, CRM Integration, Salesforce, Chief Marketing Officer, Demand Generation, Lead Scoring, Marketing Automation, Series B, Enterprise Software

    ### YOUR RESPONSE (Comma Separated Only):
    ---

    ### TEXT TO ANALYZE:
    {clean_text}
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return ""

# -------------------- LLM SUMMARY --------------------
def summarize_text(text):
    prompt = f"""
    ### ROLE
    You are a Market Research Analyst preparing a brief for a Sales Director.

    ### TASK
    Summarize the following company/product text into exactly 5 high-value bullet points that focus on **commercial viability**.

    ### INPUT TEXT
    {text}

    ### SUMMARY STRUCTURE (Strictly follow this format)
    1. **The Hook:** What is the core product/service in 1 sentence?
    2. **Target Audience:** Who specifically buys this? (Job titles or Industries).
    3. **Value Proposition:** What specific financial or operational problem does it solve?
    4. **Tech/Key Feature:** The most important technical detail or differentiator.
    5. **Revenue Model:** (Infer if possible) Is it SaaS, Agency, Physical Product, or Marketplace?

    ### OUTPUT FORMAT
    Return only the 5 bullet points. Keep them concise and professional.
    """

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except:
        return "Summary unavailable."

# -------------------- TF-IDF EXTRACTION --------------------
def extract_tfidf_keywords(text, top_n=15):
    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    sorted_words = sorted(scores, key=lambda x: x[1], reverse=True)
    return [w for w, s in sorted_words[:top_n]]
def clean_keywords(keywords):
    useless = set([
        "data", "research", "product", "industry", "company",
        "service", "technology", "platform", "solution", "system"
    ])
    cleaned = []

    for kw in keywords:
        kw = kw.strip().lower()
        if len(kw) < 3:
            continue
        if kw in useless:
            continue
        if kw.replace(" ", "").isalpha() is False:
            continue
        cleaned.append(kw)

    return list(sorted(set(cleaned)))

# Combine related keywords intelligently
def combine_keywords(keywords):
    combined = set()
    for a in keywords:
        for b in keywords:
            if a != b and (a in b or b in a):
                combined.add(f"{a} {b}")
    return list(combined)


# -------------------- TOPIC MODELING (LDA) --------------------
def extract_topics(text, n_topics=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()

    topics = []
    for topic in lda.components_:
        top_words = [words[i] for i in topic.argsort()[-5:]]
        topics.append(", ".join(top_words))

    return topics

# -------------------- NAMED ENTITY RECOGNITION --------------------
def extract_ner(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON"]]
    return list(set(entities))

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title="LeadGen Intelligence Engine", layout="wide")

if "final_keywords" not in st.session_state:
    st.session_state.final_keywords = []

st.title("📌 LeadGen Intelligence Engine")
st.markdown("Upload text/documents → Extract Keywords → Clean & Review → Save for Lead Targeting")
st.markdown("---")

# -------------------- FORM INPUTS --------------------
with st.container(border=True):
    location = st.text_input("Location")
    job_title = st.text_input("Job Title")
    industry = st.text_input("Industry")
    about = st.text_area("About (Company Description)", height=180)

    uploaded_docs = st.file_uploader("Upload Files", accept_multiple_files=True)

    submit_btn = st.button("Run Analysis", type="primary", use_container_width=True)

# -------------------- PROCESSING --------------------
summary = ""
topics = []


if submit_btn:

    all_text = about

    if uploaded_docs:
        for f in uploaded_docs:
            all_text += " " + extract_text_from_file(f)

    cleaned_text = preprocess_text(all_text)

    # LLM Keywords
    llm_keywords = extract_keywords_llm(all_text).split(",")

    # TF-IDF keywords
    tfidf_keywords = extract_tfidf_keywords(cleaned_text)

    # Topic modeling
    topics = extract_topics(cleaned_text)

    # NER entities
    ner_keywords = extract_ner(all_text)

    # Summary
    summary = summarize_text(all_text)

    # FINAL MERGED KEYWORD LIST
    combined_keywords = combine_keywords(llm_keywords)
    merged_keywords = clean_keywords(
        llm_keywords + tfidf_keywords + ner_keywords + topics + combined_keywords
    )

    st.session_state.final_keywords = merged_keywords

    st.success("✅ Analysis Complete!")
    st.balloons()

# -------------------- RESULTS DISPLAY --------------------
# -------------------- RESULTS DISPLAY --------------------
st.subheader("📌 Summary")

# Persist summary
if "summary" not in st.session_state:
    st.session_state.summary = summary
else:
    if summary:
        st.session_state.summary = summary

st.write(st.session_state.summary)

# -----------------------------------
# Deduplicate topics
# -----------------------------------
def dedupe_topics(topics_list):
    cleaned = []
    seen = set()
    for t in topics_list:
        t_clean = ", ".join(sorted(set(t.split(", "))))
        if t_clean not in seen:
            cleaned.append(t_clean)
            seen.add(t_clean)
    return cleaned

if "topics" not in st.session_state:
    st.session_state.topics = topics
else:
    if topics:
        st.session_state.topics = dedupe_topics(topics)

st.subheader("📌 Topics Identified")
st.write(st.session_state.topics)

# -----------------------------------
# Keyword selection UI
# -----------------------------------
st.subheader("📌 Manage Keywords")

if "selected_keywords" not in st.session_state:
    st.session_state.selected_keywords = set()

cols = st.columns(3)

for i, kw in enumerate(st.session_state.final_keywords):
    col = cols[i % 3]
    with col:
        checked = st.checkbox(
            kw,
            key=f"kw_{kw}",
            value=kw in st.session_state.selected_keywords
        )
        if checked:
            st.session_state.selected_keywords.add(kw)
        else:
            st.session_state.selected_keywords.discard(kw)

# Delete selected
if st.button("❌ Delete Selected Keywords", type="secondary"):
    st.session_state.final_keywords = [
        k for k in st.session_state.final_keywords
        if k not in st.session_state.selected_keywords
    ]
    st.session_state.selected_keywords = set()
    st.rerun()

# Show cleaned list
st.subheader("Final Keywords")
st.write(list(st.session_state.final_keywords))
