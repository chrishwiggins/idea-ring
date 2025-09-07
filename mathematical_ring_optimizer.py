#!/usr/bin/env python3
"""
MATHEMATICAL RING OPTIMIZER - MATH STYLE! 🔥
Implements closure-constrained TSP with advanced optimization techniques
"""
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from itertools import combinations
import random
from datetime import datetime, timedelta
import pandas as pd

class MathematicalRingOptimizer:
    def __init__(self, closure_weight=5.0, max_iterations=100):
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        self.closure_weight = closure_weight  # Penalty for poor ring closure
        self.max_iterations = max_iterations
        
    def compute_enhanced_distance_matrix(self, embeddings):
        """
        🧮 MATHEMATICAL DISTANCE MATRIX WITH CLOSURE CONSTRAINTS
        Uses weighted combination of Euclidean and cosine distances
        """
        n = len(embeddings)
        
        # Standard Euclidean distances
        euclidean_distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                euclidean_distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
        
        # Cosine distances (1 - cosine similarity)
        cosine_distances = 1 - cosine_similarity(embeddings)
        
        # Weighted combination - MATH STYLE!
        alpha = 0.7  # Weight for Euclidean
        beta = 0.3   # Weight for cosine
        combined_distances = alpha * euclidean_distances + beta * cosine_distances
        
        return combined_distances, euclidean_distances, cosine_distances
    
    def closure_constrained_tsp(self, embeddings, ideas):
        """
        🎯 CLOSURE-CONSTRAINED TSP SOLVER
        Penalizes solutions with poor semantic wrap-around
        """
        n = len(embeddings)
        distances, euclidean, cosine = self.compute_enhanced_distance_matrix(embeddings)
        
        best_path = None
        best_score = float('inf')
        best_closure = float('inf')
        
        print("🔍 SEARCHING FOR OPTIMAL STARTING POINTS...")
        
        # Try multiple starting points - MATHEMATICAL BRUTE FORCE!
        for start_idx in range(min(n, 20)):  # Limit for performance
            try:
                # Create graph for NetworkX TSP
                G = nx.complete_graph(n)
                for i in range(n):
                    for j in range(n):
                        if i != j:  # Avoid self-loops
                            G[i][j]["weight"] = distances[i, j]
                
                # Solve TSP from this starting point
                path = nx.approximation.greedy_tsp(G, source=start_idx)
                
                if len(path) < 3:
                    continue
                    
                # Calculate path cost
                path_cost = 0
                for i in range(len(path) - 1):
                    path_cost += distances[path[i], path[i + 1]]
                
                # CLOSURE PENALTY - THE MATHEMATICAL MAGIC! ✨
                closure_distance = distances[path[-2], path[0]]  # Last to first
                closure_penalty = self.closure_weight * closure_distance
                
                total_score = path_cost + closure_penalty
                
                if total_score < best_score:
                    best_score = total_score
                    best_path = path
                    best_closure = closure_distance
                    
                    print(f"  🎯 New best from start {start_idx}: closure={closure_distance:.3f}, total={total_score:.3f}")
                    
            except Exception as e:
                continue
        
        if best_path is None:
            # Fallback to simple greedy
            G = nx.complete_graph(n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        G[i][j]["weight"] = distances[i, j]
            best_path = nx.approximation.greedy_tsp(G, source=0)
            best_closure = distances[best_path[-2], best_path[0]]
        
        return best_path, best_closure, distances
    
    def two_opt_optimization(self, path, distances):
        """
        🔄 2-OPT LOCAL SEARCH OPTIMIZATION
        Classic TSP improvement with closure awareness
        """
        print("🔄 APPLYING 2-OPT OPTIMIZATION...")
        
        def calculate_total_cost(path, distances, closure_weight):
            cost = 0
            n = len(path) - 1  # Exclude duplicate end node
            for i in range(n):
                j = (i + 1) % n
                cost += distances[path[i], path[j]]
            
            # Add closure penalty
            closure_cost = closure_weight * distances[path[n-1], path[0]]
            return cost + closure_cost
        
        current_path = path[:-1]  # Remove duplicate end node
        current_cost = calculate_total_cost(current_path + [current_path[0]], distances, self.closure_weight)
        
        improved = True
        iterations = 0
        
        while improved and iterations < self.max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(current_path)):
                for j in range(i + 2, len(current_path)):
                    # Create new path by reversing segment
                    new_path = current_path[:i+1] + current_path[i+1:j+1][::-1] + current_path[j+1:]
                    new_cost = calculate_total_cost(new_path + [new_path[0]], distances, self.closure_weight)
                    
                    if new_cost < current_cost:
                        current_path = new_path
                        current_cost = new_cost
                        improved = True
                        print(f"    ⚡ Iteration {iterations}: improved cost to {current_cost:.3f}")
                        break
                        
                if improved:
                    break
        
        optimized_path = current_path + [current_path[0]]  # Add closure
        final_closure = distances[current_path[-1], current_path[0]]
        
        print(f"✅ 2-OPT completed in {iterations} iterations")
        print(f"🎯 Final closure distance: {final_closure:.3f}")
        
        return optimized_path, final_closure
    
    def create_semantic_interpolation(self, ordered_ideas, embeddings, target_slots=144):
        """
        🎨 SEMANTIC INTERPOLATION TO FILL 12 HOURS
        Creates smooth transitions between existing ideas
        """
        print(f"🎨 INTERPOLATING TO {target_slots} TIME SLOTS...")
        
        if len(ordered_ideas) >= target_slots:
            return ordered_ideas[:target_slots]
        
        # Calculate interpolation factor
        repeat_factor = target_slots / len(ordered_ideas)
        
        # Method 1: Smart repetition with clustering
        if repeat_factor <= 3:
            # Simple repetition with small variations
            full_ring = []
            cycle_count = int(repeat_factor)
            remainder = target_slots % len(ordered_ideas)
            
            for cycle in range(cycle_count):
                full_ring.extend(ordered_ideas)
            
            # Add remaining ideas
            if remainder > 0:
                full_ring.extend(ordered_ideas[:remainder])
                
        else:
            # Method 2: Clustering-based interpolation for large gaps
            # Create clusters and fill with similar ideas
            n_clusters = min(12, len(ordered_ideas) // 3)  # Up to 12 semantic clusters
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Create interpolated sequence
            full_ring = []
            slots_per_idea = target_slots // len(ordered_ideas)
            
            for i, idea in enumerate(ordered_ideas):
                full_ring.append(idea)
                
                # Add interpolated ideas in same cluster
                cluster_id = clusters[i]
                cluster_ideas = [ordered_ideas[j] for j in range(len(ordered_ideas)) 
                               if clusters[j] == cluster_id and j != i]
                
                # Add variations
                for _ in range(slots_per_idea - 1):
                    if cluster_ideas:
                        full_ring.append(random.choice(cluster_ideas))
                    else:
                        full_ring.append(f"{idea} (variation)")
            
            # Trim to exact length
            full_ring = full_ring[:target_slots]
        
        print(f"✅ Created {len(full_ring)} time slots")
        return full_ring
    
    def analyze_ring_quality(self, original_ideas, optimized_ideas, embeddings, optimized_embeddings):
        """
        📊 MATHEMATICAL ANALYSIS OF RING QUALITY
        """
        print("\n" + "="*60)
        print("📊 MATHEMATICAL RING QUALITY ANALYSIS")
        print("="*60)
        
        # Original ring analysis
        if len(original_ideas) >= 2:
            first_emb = self.model.encode([original_ideas[0]])
            last_emb = self.model.encode([original_ideas[-1]])
            original_closure = 1 - cosine_similarity(first_emb, last_emb)[0][0]
            original_avg_sim = np.mean(cosine_similarity(embeddings))
        else:
            original_closure = float('inf')
            original_avg_sim = 0
        
        # Optimized ring analysis
        opt_first_emb = self.model.encode([optimized_ideas[0]])
        opt_last_emb = self.model.encode([optimized_ideas[-1]])
        optimized_closure = 1 - cosine_similarity(opt_first_emb, opt_last_emb)[0][0]
        optimized_avg_sim = np.mean(cosine_similarity(optimized_embeddings))
        
        # Quality metrics
        print(f"🔄 CLOSURE QUALITY:")
        print(f"  Original:  {original_closure:.4f} ({'POOR' if original_closure > 0.5 else 'GOOD'})")
        print(f"  Optimized: {optimized_closure:.4f} ({'POOR' if optimized_closure > 0.5 else 'GOOD'})")
        print(f"  Improvement: {((original_closure - optimized_closure) / original_closure * 100):.1f}%")
        
        print(f"\n📏 RING COMPLETENESS:")
        print(f"  Original slots: {len(original_ideas)}")
        print(f"  Optimized slots: {len(optimized_ideas)}")
        print(f"  Coverage: {(len(optimized_ideas) / 144) * 100:.1f}% of 12 hours")
        
        print(f"\n🎯 SEMANTIC CONSISTENCY:")
        print(f"  Original avg similarity: {original_avg_sim:.4f}")
        print(f"  Optimized avg similarity: {optimized_avg_sim:.4f}")
        
        print(f"\n✨ RING TRANSITION EXAMPLES:")
        print(f"  Start → End: '{optimized_ideas[0]}' → '{optimized_ideas[-1]}'")
        for i in range(min(3, len(optimized_ideas)-1)):
            print(f"  {i+1:2}: '{optimized_ideas[i]}' → '{optimized_ideas[i+1]}'")
    
    def generate_optimized_ring(self, ideas):
        """
        🚀 MAIN OPTIMIZATION PIPELINE - MATH STYLE!
        """
        print("🚀 MATHEMATICAL RING OPTIMIZATION INITIATED!")
        print(f"📊 Processing {len(ideas)} research interests...")
        
        # Step 1: Encode all ideas
        embeddings = self.model.encode(ideas)
        print(f"✅ Generated {embeddings.shape[0]} semantic embeddings")
        
        # Step 2: Solve closure-constrained TSP
        print("\n🎯 SOLVING CLOSURE-CONSTRAINED TSP...")
        optimized_path, closure_distance, distances = self.closure_constrained_tsp(embeddings, ideas)
        
        # Step 3: Apply 2-opt optimization
        final_path, final_closure = self.two_opt_optimization(optimized_path, distances)
        
        # Step 4: Extract optimized idea sequence
        optimized_ideas = [ideas[idx] for idx in final_path[:-1]]  # Remove duplicate end
        
        # Step 5: Create full 12-hour ring
        print("\n🎨 CREATING FULL 12-HOUR SEMANTIC RING...")
        full_ring = self.create_semantic_interpolation(optimized_ideas, embeddings[final_path[:-1]])
        
        # Step 6: Generate time slots
        ring_with_times = []
        for i, idea in enumerate(full_ring):
            # 5-minute intervals starting at midnight
            minutes = (i * 5) % (12 * 60)  # Wrap at 12 hours
            hour = minutes // 60
            minute = minutes % 60
            
            # Format as 12-hour time
            if hour == 0:
                time_str = f"12:{minute:02d} AM"
            elif hour < 12:
                time_str = f"{hour:02d}:{minute:02d} AM" 
            else:
                time_str = f"12:{minute:02d} AM"  # This case shouldn't happen with 12h wrap
                
            ring_with_times.append({
                'time': time_str,
                'idea': idea,
                'index': i
            })
        
        # Step 7: Quality analysis
        optimized_embeddings = self.model.encode([item['idea'] for item in ring_with_times])
        self.analyze_ring_quality(ideas, [item['idea'] for item in ring_with_times], 
                                 embeddings, optimized_embeddings)
        
        return ring_with_times, final_closure

def load_student_data():
    """Load student research interests from the facts file"""
    facts_file = '/Users/wiggins/gd/local/Science/Teaching/Courses/Seminar/f2025/src/facts-parse/facts_data.tsv'
    
    try:
        facts_df = pd.read_csv(facts_file, sep='\t')
    except:
        facts_df = pd.read_csv(facts_file)
    
    # Extract research interests
    interest_col = facts_df.columns[-1]  # Last column should be research interests
    
    all_interests = []
    for _, row in facts_df.iterrows():
        if pd.notna(row[interest_col]):
            interests_text = str(row[interest_col])
            interests = [interest.strip() for interest in interests_text.split(',')]
            interests = [interest for interest in interests if interest and len(interest) > 2]
            all_interests.extend(interests)
    
    # Get unique interests
    unique_interests = list(set(all_interests))
    print(f"🔬 Loaded {len(unique_interests)} unique research interests")
    
    return unique_interests

def save_optimized_ring(ring_with_times, output_file):
    """Save the optimized ring to file"""
    with open(output_file, 'w') as f:
        f.write("MATHEMATICALLY OPTIMIZED RESEARCH INTEREST RING\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total slots: {len(ring_with_times)}\n")
        f.write(f"Closure quality: MATHEMATICALLY OPTIMIZED\n\n")
        
        for item in ring_with_times:
            f.write(f"{item['time']} - {item['idea']}\n")
    
    print(f"💾 Saved optimized ring to: {output_file}")

if __name__ == "__main__":
    print("🔥" * 20)
    print("MATHEMATICAL RING OPTIMIZER - INITIATED!")
    print("🔥" * 20)
    
    # Load data
    unique_interests = load_student_data()
    
    # Initialize optimizer
    optimizer = MathematicalRingOptimizer(closure_weight=10.0, max_iterations=50)
    
    # Generate optimized ring
    optimized_ring, final_closure = optimizer.generate_optimized_ring(unique_interests)
    
    # Save results
    output_file = "/tmp/IR/mathematically_optimized_ring.txt"
    save_optimized_ring(optimized_ring, output_file)
    
    print("\n🎉 MATHEMATICAL OPTIMIZATION COMPLETE!")
    print(f"✨ Final closure distance: {final_closure:.4f}")
    print(f"🎯 Ring quality: {'EXCELLENT' if final_closure < 0.3 else 'GOOD' if final_closure < 0.5 else 'NEEDS WORK'}")
    print(f"📊 Coverage: {len(optimized_ring)} time slots")