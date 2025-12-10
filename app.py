import streamlit as st
import pandas as pd
from src.preprocessor import PDFPreprocessor
from src.nlp_engine import NLPProcessor
from src.visualizer import GraphVisualizer

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Research Topic Clusterer", layout="wide")

@st.cache_data
def get_clean_data(uploaded_files):
    """
    Extracts text from uploaded files.
    Cached to prevent re-processing on every interaction.
    """
    # FIX: We pass the uploaded_files argument into the class here
    preprocessor = PDFPreprocessor(uploaded_files)
    df = preprocessor.process()
    return df

def main():
    st.title("📚 Research Literature Topic Clustering")
    st.markdown("Upload research papers (PDF/TXT) to discover hidden topic clusters using TF-IDF and K-Means.")

    # --- SIDEBAR ---
    st.sidebar.header("Data Input")
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents", 
        type=['pdf', 'txt'], 
        accept_multiple_files=True
    )

    st.sidebar.header("Model Settings")
    num_clusters = st.sidebar.slider("Number of Topics", 2, 10, 3)
    run_btn = st.sidebar.button("Analyze & Cluster")

    # --- MAIN LOGIC ---
    if run_btn and uploaded_files:
        if len(uploaded_files) < 2:
            st.error("Please upload at least 2 documents to perform clustering.")
            return

        try:
            # 1. PREPROCESSING (Cached)
            with st.spinner('Extracting text from files...'):
                # This calls the wrapper function defined above
                df = get_clean_data(uploaded_files)
                
            if df.empty:
                st.error("Could not extract text from the uploaded files.")
                return

            # 2. NLP ENGINE (TF-IDF + CLUSTERING)
            with st.spinner('Analyzing topics (TF-IDF & K-Means)...'):
                nlp = NLPProcessor(df, num_clusters)
                df_processed = nlp.run_clustering()
                keywords = nlp.get_top_keywords()

            # 3. VISUALIZATION
            with st.spinner('Generating graph...'):
                viz = GraphVisualizer(df_processed, keywords)
                
                # Layout: Graph on top, details below
                st.subheader("Topic Landscape")
                fig = viz.create_scatter_plot()
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Topic Keywords")
                    for topic, words in keywords.items():
                        st.info(f"**{topic}:** {words}")

                with col2:
                    st.subheader("Document Details")
                    st.dataframe(df_processed[["Cluster", "Filename", "Preview"]])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Uncomment the lines below if you need to see the full error in your terminal
            # import traceback
            # traceback.print_exc()

    elif run_btn and not uploaded_files:
        st.warning("Please upload files first.")

if __name__ == "__main__":
    main()