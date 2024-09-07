import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords

# Ensure NLTK punkt tokenizer and stopwords are downloaded
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Step 1: Setup argument parser to accept a file as input and a seed for reproducibility
parser = argparse.ArgumentParser(
    description="Cluster and summarize responses from a file."
)
parser.add_argument(
    "-f",
    "--file",
    default="datafile.txt",
    help="Path to the input file containing responses (default: datafile.txt)",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=42,
    help="Random seed for KMeans clustering (default: 42)",
)

# Parse the arguments
args = parser.parse_args()

# Step 2: Read the responses from the input file (or use default "datafile.txt")
with open(args.file, "r") as file:
    responses = [line.strip() for line in file.readlines() if line.strip()]

# Debug: Print the number of responses to verify the file was read correctly
print(f"Number of responses: {len(responses)}")

# Step 3: Generate embeddings for each response using a pretrained sentence transformer model
# The transformer model converts each response into a numerical vector (embedding) that captures its semantic meaning
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)  # A lightweight sentence embedding model
embeddings = model.encode(responses)  # Convert each response into a vector

# Step 4: Perform KMeans clustering on the embeddings
# n_clusters defines how many clusters we want to create (set to 8 here)
n_clusters = 8  # You can adjust this based on elbow method or other metrics
kmeans = KMeans(
    n_clusters=n_clusters, random_state=args.seed
)  # Initialize KMeans with a specified seed
kmeans.fit(embeddings)  # Perform the clustering based on the response embeddings
cluster_labels = kmeans.labels_  # Get the cluster assignment for each response

# Step 5: Calculate the overall silhouette score for the clustering
# Silhouette score ranges from -1 (bad clustering) to 1 (good clustering)
silhouette_avg = silhouette_score(embeddings, cluster_labels)
print(
    f"Average Silhouette Score for all clusters: {silhouette_avg:.2f}"
)  # Print the overall silhouette score

# Step 6: Calculate the silhouette score for each individual response
silhouette_vals = silhouette_samples(
    embeddings, cluster_labels
)  # Calculate silhouette scores for each sample

# Initialize defaultdicts to store responses and their silhouette scores per cluster
clustered_responses = defaultdict(
    list
)  # Store responses based on their cluster assignment
cluster_silhouette = defaultdict(
    list
)  # Store silhouette scores for each response based on cluster assignment

# Loop through each response, adding it to the appropriate cluster
for i, label in enumerate(cluster_labels):
    clustered_responses[label].append(
        responses[i]
    )  # Add response to corresponding cluster
    cluster_silhouette[label].append(
        silhouette_vals[i]
    )  # Add silhouette score to corresponding cluster

# Step 7: Calculate the average silhouette score for each cluster
cluster_avg_silhouette = (
    {}
)  # Dictionary to store the average silhouette score for each cluster
for cluster_id, silhouette_values in cluster_silhouette.items():
    cluster_avg_silhouette[cluster_id] = np.mean(
        silhouette_values
    )  # Average silhouette score for each cluster

# Debug: Print the average silhouette score for each cluster
print(f"Cluster silhouette scores: {cluster_avg_silhouette}")

# Step 8: Sort clusters by their average silhouette score in descending order
sorted_clusters = sorted(
    cluster_avg_silhouette.items(), key=lambda x: x[1], reverse=True
)  # Sort clusters from best to worst silhouette score

# Debug: Print the sorted cluster IDs and their average silhouette scores
print(f"Sorted clusters by silhouette score: {sorted_clusters}")


# Step 9: Function to summarize each cluster, including example responses, frequent terms, and silhouette score
def summarize_cluster(cluster_id, cluster_responses, silhouette_vals):
    print(f"\n--- Cluster {cluster_id + 1} ---")  # Print cluster ID (1-based indexing)
    print("\nCluster Summary:")

    # Print a few example responses from the cluster (up to 3 responses)
    print("\nExample Responses:")
    for response in cluster_responses[
        :3
    ]:  # Show the first 3 responses from the cluster
        print(f"- {response}")

    # Perform word frequency analysis to get the most common terms in the cluster
    all_words = nltk.word_tokenize(
        " ".join(cluster_responses).lower()
    )  # Tokenize the responses into words
    word_freq = Counter(
        [
            word for word in all_words if word.isalnum() and word not in stop_words
        ]  # Exclude stopwords and non-alphanumeric terms
    )

    # Print the top 10 common terms in the cluster
    common_words = [word for word, freq in word_freq.most_common(10)]
    print(f"\nCommon terms: {', '.join(common_words)}")

    # Calculate and print the average silhouette score for the cluster
    avg_silhouette = np.mean(silhouette_vals)
    print(f"Average Silhouette Score for this cluster: {avg_silhouette:.2f}")


# Step 10: Summarize all clusters in order of descending silhouette score
for cluster_id, _ in sorted_clusters:
    summarize_cluster(
        cluster_id, clustered_responses[cluster_id], cluster_silhouette[cluster_id]
    )  # Summarize each cluster
