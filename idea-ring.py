import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx


def extract_ideas_from_line(line):
    """
    Extract individual ideas from a given line of text.

    Args:
    - line (str): The input line containing ideas separated by delimiters.

    Returns:
    - List[str]: A list of ideas extracted from the line.
    """

    # Define the delimiters used in the line to separate ideas
    delimiters = [".", ";", ",", "/", "and"]

    # Replace each delimiter with a common delimiter ('|')
    for delimiter in delimiters:
        line = line.replace(delimiter, "|")

    # Split the line using the common delimiter and return the cleaned ideas
    return [idea.strip() for idea in line.split("|") if idea.strip()]


# Read datafile and extract unique ideas
ideas = []
with open("datafile.txt", "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        ideas.extend(extract_ideas_from_line(line))

# Convert the list of ideas into embeddings using the sentence-transformers library
# This helps in understanding the semantic similarity between different ideas
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(ideas)

# Compute pairwise similarities using Euclidean distance
# This distance will help in understanding how similar two ideas are
distances = np.zeros((len(ideas), len(ideas)))
for i in range(len(ideas)):
    for j in range(len(ideas)):
        distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

# Build a graph where nodes are ideas and edges represent distances between them
# The goal is to find connections between different ideas based on their similarities
G = nx.complete_graph(len(ideas))
for i, j in G.edges():
    G[i][j]["weight"] = distances[i][j]

# Find a path through the graph that visits each idea approximately once
# This uses a greedy algorithm for solving the Traveling Salesman Problem (TSP)
path = nx.approximation.greedy_tsp(G, source=0)

# Reorder ideas based on the computed path
# This creates an ordered sequence of ideas, showing a potential flow of thoughts
clock_time_ideas = [ideas[idx] for idx in path]

# Calculate the interval between each idea based on the total number of ideas
# This ensures each idea gets a unique timestamp within a 12-hour period
total_ideas = len(clock_time_ideas)
minutes_interval = (12 * 60) // total_ideas

# Assign a unique timestamp to each idea and print the results
# This creates a schedule or a timeline for exploring these ideas
current_minutes = 0
for idea in clock_time_ideas:
    hour = (current_minutes // 60) % 12
    if hour == 0:
        hour = 12
    minute = current_minutes % 60
    print(f"{hour:02}:{minute:02} - {idea}")
    current_minutes += minutes_interval
