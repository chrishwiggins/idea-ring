#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import seaborn as sns

def extract_research_interests(facts_df):
    """Extract research interests from facts dataframe"""
    # Find the research interests column
    interest_col = None
    for col in facts_df.columns:
        if 'idea' in col.lower() and 'research' in col.lower():
            interest_col = col
            break
    
    if not interest_col:
        # Use the last column as fallback
        interest_col = facts_df.columns[-1]
    
    print(f"Extracting interests from: '{interest_col}'")
    
    # Extract all research interests with student attribution
    all_interests = []
    student_interests = {}
    interest_to_students = defaultdict(list)
    
    for _, row in facts_df.iterrows():
        uni_col = 'uni (all lowercase, no special characters, no spaces)'
        if pd.notna(row[interest_col]) and uni_col in facts_df.columns:
            interests_text = str(row[interest_col])
            interests = [interest.strip() for interest in interests_text.split(',')]
            interests = [interest for interest in interests if interest and len(interest) > 2]
            
            student_interests[row[uni_col]] = interests
            all_interests.extend(interests)
            
            for interest in interests:
                interest_to_students[interest].append(row[uni_col])
    
    return all_interests, student_interests, interest_to_students

def create_idea_ring(ideas, interest_to_students, category_filter=None, ring_hours=12):
    """Create semantic idea ring with TSP optimization"""
    
    if category_filter:
        print(f"Filtering for category: {category_filter}")
        # Add category filtering logic here if needed
    
    unique_ideas = list(set(ideas))
    print(f"Processing {len(unique_ideas)} unique ideas...")
    
    # Create embeddings
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(unique_ideas)
    
    # Compute pairwise distances
    distances = np.zeros((len(unique_ideas), len(unique_ideas)))
    for i in range(len(unique_ideas)):
        for j in range(len(unique_ideas)):
            distances[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
    
    # Build graph
    G = nx.complete_graph(len(unique_ideas))
    for i, j in G.edges():
        G[i][j]["weight"] = distances[i][j]
    
    # Find optimal path
    if len(unique_ideas) > 1:
        path = nx.approximation.greedy_tsp(G, source=0)
        ordered_ideas = [unique_ideas[idx] for idx in path[:-1]]  # Remove duplicate end node
    else:
        ordered_ideas = unique_ideas
    
    # Assign clock positions
    total_ideas = len(ordered_ideas)
    minutes_interval = (ring_hours * 60) // total_ideas if total_ideas > 0 else 1
    
    clock_positions = []
    current_minutes = 0
    
    for idea in ordered_ideas:
        hour = (current_minutes // 60) % ring_hours
        minute = current_minutes % 60
        
        if ring_hours == 12:
            period = "AM" if current_minutes < (12 * 60) else "PM"
            if hour == 0:
                hour = 12
            time_str = f"{hour:02}:{minute:02} {period}"
        else:
            time_str = f"{hour:02}:{minute:02}"
        
        students = interest_to_students.get(idea, [])
        clock_positions.append({
            'time': time_str,
            'idea': idea,
            'students': students,
            'student_count': len(students)
        })
        
        current_minutes += minutes_interval
    
    return clock_positions, G, embeddings, distances

def visualize_physical_ring(clock_positions, output_file=None):
    """Create actual circular ring visualization for physical room arrangement"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left plot: Circular ring arrangement
    n_ideas = len(clock_positions)
    angles = np.linspace(0, 2*np.pi, n_ideas, endpoint=False)
    
    # Create circle positions
    radius = 1
    x_pos = radius * np.cos(angles)
    y_pos = radius * np.sin(angles)
    
    # Plot the ring
    ax1.scatter(x_pos, y_pos, s=200, c='lightblue', alpha=0.7, edgecolors='navy')
    
    # Add labels with idea names and times
    for i, (pos, angle) in enumerate(zip(clock_positions, angles)):
        idea_short = pos['idea'][:25] + ('...' if len(pos['idea']) > 25 else '')
        
        # Position labels outside the circle
        label_radius = 1.3
        label_x = label_radius * np.cos(angle)
        label_y = label_radius * np.sin(angle)
        
        # Adjust text alignment based on position
        ha = 'left' if label_x > 0 else 'right'
        va = 'bottom' if label_y > 0 else 'top'
        if abs(label_x) < 0.1:  # Near vertical axis
            ha = 'center'
        if abs(label_y) < 0.1:  # Near horizontal axis  
            va = 'center'
        
        ax1.annotate(f"{pos['time']}\n{idea_short}", 
                    xy=(x_pos[i], y_pos[i]), 
                    xytext=(label_x, label_y),
                    ha=ha, va=va, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    # Draw clock face
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2)
    ax1.add_patch(circle)
    
    # Add clock hour markers
    hour_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for i, angle in enumerate(hour_angles):
        hour = i if i != 0 else 12
        marker_x = 0.9 * np.cos(angle)
        marker_y = 0.9 * np.sin(angle)
        ax1.text(marker_x, marker_y, str(hour), ha='center', va='center', 
                fontsize=12, fontweight='bold', 
                bbox=dict(boxstyle="circle,pad=0.1", facecolor='yellow', alpha=0.7))
    
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_title('Physical Ring Arrangement\n(Students sit at their topic times)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right plot: Student-Topic mapping
    student_topics = defaultdict(list)
    for pos in clock_positions:
        for student in pos['students']:
            student_topics[student].append((pos['time'], pos['idea']))
    
    # Create student mapping text
    y_offset = 0.95
    ax2.text(0.05, y_offset, "Student-Topic Assignments:", fontsize=14, fontweight='bold', transform=ax2.transAxes)
    y_offset -= 0.05
    
    for student, topics in sorted(student_topics.items()):
        y_offset -= 0.04
        if y_offset < 0.05:
            break  # Prevent overflow
        
        topic_str = ", ".join([f"{time}" for time, topic in topics])
        ax2.text(0.05, y_offset, f"{student}: {topic_str}", fontsize=10, 
                transform=ax2.transAxes, family='monospace')
    
    ax2.axis('off')
    ax2.set_title('Student Positions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Ring visualization saved as: {output_file}")
    
    plt.show()

def visualize_network(G, ideas, embeddings, output_file=None):
    """Create properly labeled network visualization"""
    
    plt.figure(figsize=(16, 12))
    
    # Use circular layout for better visibility
    pos = nx.circular_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
    
    # Draw only the strongest connections to avoid clutter
    all_weights = [d['weight'] for u, v, d in G.edges(data=True)]
    threshold = np.percentile(all_weights, 15)  # Show top 15% strongest connections
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, alpha=0.3, width=1.5, edge_color='red')
    
    # Add ALL labels - this was the main issue!
    labels = {}
    for i, idea in enumerate(ideas):
        # Truncate very long ideas but keep them readable
        if len(idea) > 30:
            words = idea.split()
            if len(words) > 4:
                labels[i] = ' '.join(words[:4]) + '...'
            else:
                labels[i] = idea[:30] + '...'
        else:
            labels[i] = idea
    
    # Draw labels with better positioning
    for node, (x, y) in pos.items():
        plt.text(x, y, labels[node], fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'),
                wrap=True)
    
    plt.title("Research Interest Network\n(Red edges = strongest semantic connections)", fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Network visualization saved as: {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Create idea ring from facts data")
    parser.add_argument("facts_file", help="Path to facts TSV/CSV file")
    parser.add_argument("--category", help="Filter by category (ML, Biology, Physics, etc.)")
    parser.add_argument("--hours", type=int, default=12, help="Number of hours for the ring (default: 12)")
    parser.add_argument("--viz", action="store_true", help="Create network visualization")
    parser.add_argument("--output", help="Output file prefix for results")
    
    args = parser.parse_args()
    
    # Load facts data
    try:
        facts_df = pd.read_csv(args.facts_file, sep='\t')
    except:
        facts_df = pd.read_csv(args.facts_file)
    
    print(f"Loaded {len(facts_df)} student submissions")
    
    # Extract research interests
    all_interests, student_interests, interest_to_students = extract_research_interests(facts_df)
    
    print(f"Found {len(all_interests)} total research interests from {len(student_interests)} students")
    
    # Create idea ring
    clock_positions, G, embeddings, distances = create_idea_ring(
        all_interests, interest_to_students, args.category, args.hours
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"RESEARCH INTEREST RING ({args.hours}-hour format)")
    print(f"{'='*60}")
    
    for pos in clock_positions:
        students_str = ", ".join(pos['students']) if len(pos['students']) <= 3 else f"{', '.join(pos['students'][:3])}... ({pos['student_count']} total)"
        print(f"{pos['time']} - {pos['idea']}")
        if pos['students']:
            print(f"         Students: {students_str}")
        print()
    
    # Save results
    if args.output:
        output_file = f"{args.output}_ring.txt"
        with open(output_file, 'w') as f:
            f.write(f"Research Interest Ring ({args.hours}-hour format)\n")
            f.write("="*60 + "\n\n")
            for pos in clock_positions:
                f.write(f"{pos['time']} - {pos['idea']}\n")
                if pos['students']:
                    students_str = ", ".join(pos['students'])
                    f.write(f"         Students: {students_str}\n")
                f.write("\n")
        print(f"Ring saved to: {output_file}")
    
    # Create visualizations
    if args.viz:
        # Physical ring arrangement
        ring_file = f"{args.output}_ring.png" if args.output else "research_ring.png"
        visualize_physical_ring(clock_positions, ring_file)
        
        # Network graph  
        network_file = f"{args.output}_network.png" if args.output else "research_network.png"
        visualize_network(G, list(set(all_interests)), embeddings, network_file)
    
    print(f"\nSummary: {len(clock_positions)} unique research interests arranged in semantic order")
    print(f"Use this arrangement for student discussion groups!")

if __name__ == "__main__":
    main()