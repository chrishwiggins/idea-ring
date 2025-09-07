import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import argparse

# Argument parser to allow optional file input
parser = argparse.ArgumentParser(description="Transform brainstormed ideas into an optimized timeline using AI and mathematical optimization.")
parser.add_argument(
    "-f", "--file", default="datafile.txt", help="Path to the input file (default: datafile.txt)"
)
parser.add_argument(
    "--closure-weight", type=float, default=5.0, help="Weight for ring closure penalty (default: 5.0)"
)
parser.add_argument(
    "--optimize", action="store_true", help="Enable advanced TSP optimization with 2-opt local search"
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

def compute_enhanced_distance_matrix(embeddings):
    """
    Compute enhanced distance matrix using weighted combination of Euclidean and cosine distances.
    
    This hybrid approach combines:
    - Euclidean distance: captures semantic magnitude differences
    - Cosine distance: captures semantic direction differences
    
    Args:
        embeddings: numpy array of sentence embeddings
    
    Returns:
        numpy array: combined distance matrix
    """
    n = len(embeddings)
    
    # Compute Euclidean distances
    euclidean_distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            euclidean_distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Compute cosine distances (1 - cosine similarity)
    cosine_distances = 1 - cosine_similarity(embeddings)
    
    # Weighted combination: 70% Euclidean + 30% cosine
    # This balance prioritizes semantic magnitude while considering direction
    alpha, beta = 0.7, 0.3
    combined_distances = alpha * euclidean_distances + beta * cosine_distances
    
    return combined_distances

def find_optimal_tsp_path(embeddings, ideas, closure_weight=5.0):
    """
    Find optimal TSP path with closure constraint and multiple starting points.
    
    The closure constraint penalizes solutions where the last idea doesn't
    connect well semantically with the first idea, ensuring a smooth
    "ring" structure for presentation.
    
    Args:
        embeddings: sentence embeddings for semantic similarity
        ideas: list of idea strings
        closure_weight: penalty factor for poor ring closure
    
    Returns:
        tuple: (optimal_path, closure_distance)
    """
    n = len(embeddings)
    distances = compute_enhanced_distance_matrix(embeddings)
    
    best_path = None
    best_score = float('inf')
    best_closure = float('inf')
    
    print(f"Optimizing path through {n} ideas with closure constraint...")
    
    # Try multiple starting points to find globally optimal solution
    num_trials = min(n, 20)  # Limit trials for performance
    
    for start_idx in range(num_trials):
        try:
            # Create complete graph for NetworkX TSP solver
            G = nx.complete_graph(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G[i][j]["weight"] = distances[i, j]
            
            # Solve TSP from this starting point
            path = nx.approximation.greedy_tsp(G, source=start_idx)
            
            if len(path) < 3:
                continue
            
            # Calculate total path cost
            path_cost = sum(distances[path[i], path[i + 1]] for i in range(len(path) - 1))
            
            # Add closure penalty: penalize poor semantic connection from end to start
            closure_distance = distances[path[-2], path[0]]  # Last meaningful to first
            closure_penalty = closure_weight * closure_distance
            total_score = path_cost + closure_penalty
            
            if total_score < best_score:
                best_score = total_score
                best_path = path
                best_closure = closure_distance
                print(f"  New best path from start {start_idx}: closure={closure_distance:.3f}")
        
        except Exception:
            continue
    
    if best_path is None:
        # Fallback to simple greedy solution
        G = nx.complete_graph(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i][j]["weight"] = distances[i, j]
        best_path = nx.approximation.greedy_tsp(G, source=0)
        best_closure = distances[best_path[-2], best_path[0]]
    
    print(f"Optimal path found with closure distance: {best_closure:.3f}")
    return best_path, best_closure

def two_opt_optimization(path, embeddings, closure_weight=5.0, max_iterations=50):
    """
    Apply 2-opt local search optimization to improve TSP solution.
    
    2-opt works by repeatedly reversing segments of the path to find
    improvements. This is a classic TSP optimization technique that
    can significantly improve solution quality.
    
    Args:
        path: initial TSP path
        embeddings: sentence embeddings for distance calculation
        closure_weight: penalty factor for ring closure
        max_iterations: maximum optimization iterations
    
    Returns:
        tuple: (optimized_path, final_closure_distance)
    """
    distances = compute_enhanced_distance_matrix(embeddings)
    
    def calculate_total_cost(path_segment, distances, closure_weight):
        """Calculate total cost including closure penalty."""
        cost = sum(distances[path_segment[i], path_segment[(i + 1) % len(path_segment)]] 
                  for i in range(len(path_segment)))
        
        # Add closure penalty
        closure_cost = closure_weight * distances[path_segment[-1], path_segment[0]]
        return cost + closure_cost
    
    current_path = path[:-1]  # Remove duplicate end node for processing
    current_cost = calculate_total_cost(current_path, distances, closure_weight)
    
    print(f"Starting 2-opt optimization from cost: {current_cost:.3f}")
    
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try all possible 2-opt swaps
        for i in range(len(current_path)):
            for j in range(i + 2, len(current_path)):
                # Create new path by reversing segment between i+1 and j
                new_path = (current_path[:i+1] + 
                           current_path[i+1:j+1][::-1] + 
                           current_path[j+1:])
                
                new_cost = calculate_total_cost(new_path, distances, closure_weight)
                
                if new_cost < current_cost:
                    current_path = new_path
                    current_cost = new_cost
                    improved = True
                    print(f"  Iteration {iterations}: improved cost to {current_cost:.3f}")
                    break
            
            if improved:
                break
    
    # Add closure back to path
    optimized_path = current_path + [current_path[0]]
    final_closure = distances[current_path[-1], current_path[0]]
    
    print(f"2-opt completed after {iterations} iterations")
    print(f"Final closure distance: {final_closure:.3f}")
    
    return optimized_path, final_closure

# Read datafile and extract unique ideas
ideas = []
with open(args.file, "r") as file:
    lines = file.readlines()[1:]  # Skip the header
    for line in lines:
        ideas.extend(extract_ideas_from_line(line))

print(f"Processing {len(ideas)} ideas from {args.file}")

# Convert ideas into semantic embeddings using sentence transformers
print("Generating semantic embeddings...")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
embeddings = model.encode(ideas)

# Find optimal path through ideas using advanced TSP optimization
if args.optimize:
    print("Using advanced optimization with 2-opt local search...")
    path, closure_distance = find_optimal_tsp_path(embeddings, ideas, args.closure_weight)
    path, final_closure = two_opt_optimization(path, embeddings, args.closure_weight)
else:
    print("Using basic TSP optimization...")
    path, closure_distance = find_optimal_tsp_path(embeddings, ideas, args.closure_weight)

# Reorder ideas based on the computed path
clock_time_ideas = [ideas[idx] for idx in path]

# Calculate optimal time distribution across 12-hour period
# Remove duplicate end node for display but keep count accurate
unique_ideas = clock_time_ideas[:-1]  # Remove duplicate end node
total_unique_ideas = len(unique_ideas)

# Calculate interval to fill exactly 12 hours (720 minutes)
total_minutes = 12 * 60  # 720 minutes from 12:00 AM to 12:00 PM
minutes_interval = total_minutes / total_unique_ideas  # Use float division for precise spacing

print(f"\nOptimized Idea Ring ({total_unique_ideas} unique ideas):")
print("=" * 50)

# Assign timestamps and display the optimized ring
for i, idea in enumerate(unique_ideas):
    current_minutes = i * minutes_interval  # Precise floating point calculation
    
    hour = int(current_minutes // 60) % 12
    minute = int(current_minutes % 60)
    period = "AM" if current_minutes < (12 * 60) else "PM"

    # Display 12:XX instead of 0:XX
    if hour == 0:
        hour = 12

    print(f"{hour:02d}:{minute:02d} {period} - {idea}")

# Show the wrap-around connection
print("=" * 50)
print(f"Ring closes: {unique_ideas[-1]} → {unique_ideas[0]}")
print(f"Total span: 12:00 AM to 12:00 PM (complete 12-hour cycle)")

print("=" * 50)
print(f"Ring optimization complete. Closure quality: {closure_distance:.3f}")
if args.optimize:
    print("Advanced 2-opt optimization applied for improved solution quality.")