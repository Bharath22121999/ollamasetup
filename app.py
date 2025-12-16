import streamlit as st
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# SETUP PATHS
# ---------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'my_local_model')
CSV_PATH = os.path.join(SCRIPT_DIR, 'dataset.csv')
PKL_PATH = os.path.join(SCRIPT_DIR, 'corpus_embeddings.pkl')

# ---------------------------------------------------------
# 1. LOAD MODEL (Cached so it doesn't reload every time)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    return SentenceTransformer(MODEL_PATH)

st.title("🔍 Local Semantic Search Tool")

# Load model immediately
model = load_model()

# ---------------------------------------------------------
# 2. SIDEBAR - FILE UPLOAD & SETTINGS
# ---------------------------------------------------------
st.sidebar.header("1. Update Database")
uploaded_file = st.sidebar.file_uploader("Upload new dataset.csv", type="csv")

if uploaded_file is not None:
    # A. Read and Overwrite CSV
    df = pd.read_csv(uploaded_file)
    
    # Save to disk (Replacing old file)
    df.to_csv(CSV_PATH, index=False)
    st.sidebar.success("✅ CSV File Updated!")

    # B. Ask user which column to use
    column_name = st.sidebar.selectbox("Select the Text Column:", df.columns)

    # C. Button to Regenerate Embeddings
    if st.sidebar.button("⚠️ Process & Re-Index Data"):
        with st.spinner('Encoding sentences... This may take a minute...'):
            # 1. Get List
            dataset = df[column_name].dropna().tolist()
            
            # 2. Encode
            embeddings = model.encode(dataset, convert_to_tensor=True)
            
            # 3. Save Pickle (Replacing old file)
            with open(PKL_PATH, 'wb') as f:
                pickle.dump({'sentences': dataset, 'embeddings': embeddings}, f)
                
        st.sidebar.success(f"✅ Done! Indexed {len(dataset)} rows.")

st.sidebar.markdown("---")
st.sidebar.header("2. Search Settings")
top_k = st.sidebar.slider("Number of Results (Top K)", min_value=1, max_value=50, value=10)
threshold = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# ---------------------------------------------------------
# 3. MAIN INTERFACE - SEARCH
# ---------------------------------------------------------
st.subheader("Search your Data")

# Input Box
query = st.text_input("Enter your query here:", placeholder="e.g., I lost my credit card")

if query:
    if not os.path.exists(PKL_PATH):
        st.error("⚠️ No database found! Please upload a CSV in the sidebar first.")
    else:
        # Load Data
        with open(PKL_PATH, 'rb') as f:
            data_store = pickle.load(f)
            
        stored_sentences = data_store['sentences']
        stored_embeddings = data_store['embeddings']

        # Perform Search
        query_embedding = model.encode(query, convert_to_tensor=True)
        search_results = util.semantic_search(query_embedding, stored_embeddings, top_k=top_k)
        
        # Process Results
        hits = search_results[0]
        results_list = []
        
        for hit in hits:
            score = hit['score']
            if score >= threshold:
                results_list.append({
                    "Score": f"{score:.4f}",
                    "Sentence": stored_sentences[hit['corpus_id']]
                })
        
        # Display Results
        if results_list:
            st.write(f"Found **{len(results_list)}** matches:")
            # Show as a clean interactive table
            st.dataframe(pd.DataFrame(results_list), use_container_width=True)
        else:
            st.warning("No matches found above the selected threshold.")

# Footer
st.markdown("---")
st.caption("Local Offline AI • All-MiniLM-L6-v2")
