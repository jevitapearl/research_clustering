import plotly.express as px

class GraphVisualizer:
    def __init__(self, df, keywords):
        self.df = df
        self.keywords = keywords

    def create_scatter_plot(self):
        # Add a column for hover text that includes keywords
        # We map the keywords to the cluster column
        keyword_map = {k: v for k, v in self.keywords.items()}
        self.df['Top Keywords'] = self.df['Cluster'].map(keyword_map)

        fig = px.scatter(
            self.df, 
            x="x", 
            y="y", 
            color="Cluster",
            hover_data=["Filename", "Top Keywords"],
            title="Document Cluster Map",
            template="plotly_white",
            size_max=12
        )
        
        fig.update_layout(
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            legend_title="Topics"
        )
        
        return fig