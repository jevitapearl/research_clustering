import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class NLPProcessor:
    def __init__(self, df, num_clusters):
        self.df = df
        self.num_clusters = num_clusters
        self.vectorizer = None
        self.tfidf_matrix = None
        self.kmeans = None

    def run_clustering(self):
        """Performs TF-IDF, K-Means, and PCA."""
        
        # 1. TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['Content'])

        # 2. Clustering
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(self.tfidf_matrix)
        clusters = self.kmeans.labels_

        # 3. PCA for Visualization (2D coordinates)
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(self.tfidf_matrix.toarray())

        # Update DataFrame
        self.df['Cluster_ID'] = clusters
        self.df['Cluster'] = [f"Topic {c+1}" for c in clusters]
        self.df['x'] = pca_coords[:, 0]
        self.df['y'] = pca_coords[:, 1]
        self.df['Preview'] = self.df['Content'].apply(lambda x: x[:100] + "...")
        
        return self.df

    def get_top_keywords(self, n_terms=5):
        """Extracts top keywords for each cluster."""
        if self.tfidf_matrix is None or self.kmeans is None:
            return {}

        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF vector for each cluster
        # Note: We need to group the original matrix by cluster labels
        dense_matrix = self.tfidf_matrix.todense()
        df_tfidf = pd.DataFrame(dense_matrix)
        df_tfidf['label'] = self.kmeans.labels_
        
        # Group by label and get mean
        cluster_means = df_tfidf.groupby('label').mean()
        
        keywords = {}
        for cluster_id, row in cluster_means.iterrows():
            # Get top indices
            top_indices = row.sort_values(ascending=False).head(n_terms).index
            # Convert indices to words
            words = [feature_names[i] for i in top_indices]
            keywords[f"Topic {cluster_id+1}"] = ", ".join(words)
            
        return keywords