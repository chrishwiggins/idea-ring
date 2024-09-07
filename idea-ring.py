import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
import argparse

# Argument parser to allow optional file input
parser = argparse.ArgumentParser(description="Process ideas from an input file.")
parser.add_argument(
    "-f", "--file", default="datafile.txt", help="Path to the input file (default: datafile.txt)"
)
args = parser.parse_args()

def extract_ideas_from_line(line):
    """
    Extract individual ideas from a given line of text.

    Args:
    - line (str): The input line containing ideas separated by delimiters.

    Returns:
    - List[str]: A list of ideas extracted from the line.
    """
    delimiters = [".", ";", ",", "/", "and"]
    for delimiter in delimiters:
        line = line.replace(delimiter, "|")
    return [idea.strip() for idea in line.split("|") if idea.strip()]

# Read datafile and extract unique ideas
ideas = []
with open(args.file, "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        ideas.extend(extract_ideas_from_line(line))

# Convert the list of ideas into embeddings using the sentence-transformers library
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(ideas)

# Compute pairwise similarities using Euclidean distance
distances = np.zeros((len(ideas), len(ideas)))
for i in range(len(ideas)):
    for j in range(len(ideas)):
        distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

# Build a graph where nodes are ideas and edges represent distances between them
G = nx.complete_graph(len(ideas))
for i, j in G.edges():
    G[i][j]["weight"] = distances[i][j]

# Find a path through the graph that visits each idea approximately once
path = nx.approximation.greedy_tsp(G, source=0)

# Reorder ideas based on the computed path
clock_time_ideas = [ideas[idx] for idx in path]

# Calculate the interval between each idea based on the total number of ideas (from 00:01 to 11:59)
total_ideas = len(clock_time_ideas)
minutes_interval = (12 * 60) // total_ideas

# Assign a unique timestamp to each idea and print the results
current_minutes = 0  # Start at 00:00
for idea in clock_time_ideas:
    hour = (current_minutes // 60) % 12
    minute = current_minutes % 60
    period = "AM" if current_minutes < (12 * 60) else "PM"  # AM for first 12 hours, PM for the next

    # Adjust hour to display 12:00 correctly
    if hour == 0:
        hour = 12

    print(f"{hour:02}:{minute:02} {period} - {idea}")
    
    current_minutes += minutes_interval