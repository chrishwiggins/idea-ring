# Idea-Ring Project

## Overview
Welcome to the Idea-Ring project! At the intersection of Natural Language Processing (NLP), Graph Theory, and AI, this tool is designed to create a dynamic, interconnected map of mathematical ideas, concepts, and theories sourced from student brainstorming sessions.

## How It Works
- **Idea Extraction**: the tool parses through `datafile.txt`, a collection of raw ideas generated by students. It identifies and extracts individual ideas, ensuring that even compound sentences are split into singular, focused concepts.

- **Semantic Connections**: Leveraging the `SentenceTransformer` from Hugging Face's library, each idea is encoded into a high-dimensional vector space. This process, rooted in advanced NLP, translates textual ideas into numerical embeddings, capturing the nuanced semantic relationships between different concepts.

- **Idea Mapping**: The core algorithm constructs a complete graph where each node represents an idea. Edges between nodes are weighted based on the Euclidean distance between their corresponding semantic embeddings. This method creates a 'semantic constellation', grouping ideas based on their conceptual proximity.

- **Navigating Ideas**: To traverse this complex network, the tool applies a Greedy approach to the Traveling Salesman Problem (TSP). It aims to find a Hamiltonian cycle that visits each idea-node exactly once, resulting in an ordered sequence of ideas. This sequence represents a coherent path through the landscape of interconnected concepts, enabling users to explore ideas in a logically connected manner.

- **Time Mapping**: Each idea is assigned a specific timestamp, effectively creating a 12-hour idea so students can form a ring around a room for initial group formation

## Applications
It can serve as a brainstorming aid, an educational resource, or a starting point for interdisciplinary research, particularly in fields where mathematical concepts intersect with other domains.

