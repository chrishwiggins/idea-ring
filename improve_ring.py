#!/usr/bin/env python3
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def improve_tsp_ring(ideas, embeddings):
    """Improve the TSP to create a proper circular ring with semantic closure"""
    
    # Method 1: Hamiltonian cycle with closure constraint
    def create_closure_graph(embeddings):
        """Create graph where we explicitly connect first and last nodes"""
        n = len(embeddings)
        distances = np.zeros((n, n))
        
        # Compute all pairwise distances
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
        
        return distances
    
    # Method 2: Find optimal starting point for better closure
    def find_best_starting_point(distances):
        """Find starting point that gives best ring closure"""
        n = len(distances)
        best_start = 0
        best_closure_score = float('inf')
        
        for start in range(n):
            # Create TSP path from this starting point
            G = nx.complete_graph(n)
            for i in range(n):
                for j in range(n):
                    G[i][j]["weight"] = distances[i][j]
            
            try:
                path = nx.approximation.greedy_tsp(G, source=start)
                # Check closure quality (distance from last to first)
                if len(path) > 2:
                    closure_distance = distances[path[-2]][path[0]]  # path[-1] is duplicate of path[0]
                    if closure_distance < best_closure_score:
                        best_closure_score = closure_distance
                        best_start = start
            except:
                continue
                
        return best_start, best_closure_score
    
    # Method 3: 2-opt improvement for better ring
    def two_opt_improve(path, distances):
        """Improve TSP path using 2-opt swaps"""
        def calculate_path_length(path, distances):
            length = 0
            for i in range(len(path)):
                j = (i + 1) % len(path)
                length += distances[path[i]][path[j]]
            return length
        
        improved = True
        current_path = path[:-1]  # Remove duplicate end node
        
        while improved:
            improved = False
            for i in range(len(current_path)):
                for j in range(i + 2, len(current_path)):
                    # Try swapping edges
                    new_path = current_path[:i+1] + current_path[i+1:j+1][::-1] + current_path[j+1:]
                    
                    if calculate_path_length(new_path, distances) < calculate_path_length(current_path, distances):
                        current_path = new_path
                        improved = True
                        break
                if improved:
                    break
        
        return current_path + [current_path[0]]  # Add closure
    
    # Execute improvements
    distances = create_closure_graph(embeddings)
    best_start, closure_score = find_best_starting_point(distances)
    
    # Create improved TSP
    G = nx.complete_graph(len(embeddings))
    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            G[i][j]["weight"] = distances[i][j]
    
    path = nx.approximation.greedy_tsp(G, source=best_start)
    improved_path = two_opt_improve(path, distances)
    
    # Analyze closure quality
    first_idea = ideas[improved_path[0]]
    last_idea = ideas[improved_path[-2]]  # -2 because -1 is duplicate
    closure_distance = distances[improved_path[-2]][improved_path[0]]
    
    print(f"Ring Closure Analysis:")
    print(f"Starting idea: {first_idea}")
    print(f"Ending idea: {last_idea}")
    print(f"Semantic closure distance: {closure_distance:.3f}")
    print(f"Average inter-idea distance: {np.mean(distances):.3f}")
    print(f"Closure quality: {'GOOD' if closure_distance < np.mean(distances) else 'POOR'}")
    
    return improved_path, closure_distance

def create_full_12_hour_ring(ideas, target_slots=144):
    """Create ring that fills all 12 hours with 5-minute intervals"""
    
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(ideas)
    
    # Get improved ring order
    improved_path, closure_score = improve_tsp_ring(ideas, embeddings)
    ordered_ideas = [ideas[idx] for idx in improved_path[:-1]]  # Remove duplicate
    
    # Method 1: Repeat ideas to fill 12 hours
    if len(ordered_ideas) < target_slots:
        # Calculate how many times to repeat the cycle
        repeats = target_slots // len(ordered_ideas)
        remainder = target_slots % len(ordered_ideas)
        
        full_ring = ordered_ideas * repeats + ordered_ideas[:remainder]
    else:
        full_ring = ordered_ideas[:target_slots]
    
    # Method 2: Interpolate similar ideas between existing ones
    # (More sophisticated approach)
    
    return full_ring, closure_score

# Test with your data
def analyze_current_ring():
    """Analyze the current ring from your file"""
    ring_file = '/tmp/IR/student_research_ring.txt'
    
    ideas = []
    times = []
    with open(ring_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if ' - ' in line and any(char.isdigit() for char in line.split(' - ')[0]):
            time_part, idea_part = line.split(' - ', 1)
            ideas.append(idea_part)
            times.append(time_part)
    
    print(f"Current ring analysis:")
    print(f"Number of ideas: {len(ideas)}")
    print(f"First time: {times[0] if times else 'None'}")
    print(f"Last time: {times[-1] if times else 'None'}")
    print(f"First idea: {ideas[0] if ideas else 'None'}")
    print(f"Last idea: {ideas[-1] if ideas else 'None'}")
    
    # Check semantic similarity between first and last
    if len(ideas) >= 2:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        first_emb = model.encode([ideas[0]])
        last_emb = model.encode([ideas[-1]])
        
        similarity = cosine_similarity(first_emb, last_emb)[0][0]
        distance = np.linalg.norm(first_emb - last_emb)
        
        print(f"First-Last similarity: {similarity:.3f} (1.0 = identical)")
        print(f"First-Last distance: {distance:.3f}")
        
        # Compare to average similarity
        all_embs = model.encode(ideas)
        avg_similarity = np.mean(cosine_similarity(all_embs))
        print(f"Average similarity: {avg_similarity:.3f}")
        print(f"Closure quality: {'GOOD' if similarity > avg_similarity else 'POOR'}")
        
        # Suggest improvements
        print(f"\nSuggested improvements:")
        print(f"1. Use 2-opt TSP improvement")
        print(f"2. Try different starting points") 
        print(f"3. Add closure constraint to TSP")
        print(f"4. Fill 12 hours with 5-minute intervals (144 total)")
    
    return ideas

if __name__ == "__main__":
    analyze_current_ring()