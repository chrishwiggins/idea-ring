#!/usr/bin/env python3
"""
RING COMPARISON ANALYZER - MATH STYLE! 📊
Compares original vs optimized rings with detailed analysis
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_original_ring():
    """Load the original ring data"""
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
    
    return ideas, times

def load_optimized_ring():
    """Load the optimized ring data"""
    ring_file = '/tmp/IR/mathematically_optimized_ring.txt'
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
    
    return ideas, times

def analyze_closure_quality(ideas, title):
    """Analyze semantic closure quality"""
    if len(ideas) < 2:
        return 0, 0, []
    
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(ideas)
    
    # Calculate closure similarity (first to last)
    first_emb = embeddings[0:1]
    last_emb = embeddings[-1:]
    closure_similarity = cosine_similarity(first_emb, last_emb)[0][0]
    
    # Calculate average pairwise similarity
    similarity_matrix = cosine_similarity(embeddings)
    # Remove diagonal (self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarity = np.mean(similarity_matrix)
    
    # Calculate transition smoothness (adjacent ideas similarity)
    transition_similarities = []
    for i in range(len(ideas) - 1):
        sim = cosine_similarity(embeddings[i:i+1], embeddings[i+1:i+2])[0][0]
        transition_similarities.append(sim)
    
    # Add wrap-around transition
    wrap_sim = cosine_similarity(embeddings[-1:], embeddings[0:1])[0][0]
    transition_similarities.append(wrap_sim)
    
    print(f"\n🔍 ANALYSIS: {title}")
    print(f"   Total ideas: {len(ideas)}")
    print(f"   First idea: '{ideas[0]}'")
    print(f"   Last idea: '{ideas[-1]}'")
    print(f"   Closure similarity: {closure_similarity:.4f}")
    print(f"   Average similarity: {avg_similarity:.4f}")
    print(f"   Avg transition similarity: {np.mean(transition_similarities[:-1]):.4f}")
    print(f"   Wrap-around similarity: {wrap_sim:.4f}")
    print(f"   Closure quality: {'EXCELLENT' if closure_similarity > 0.3 else 'GOOD' if closure_similarity > 0.15 else 'POOR'}")
    
    return closure_similarity, avg_similarity, transition_similarities

def create_ring_visualization():
    """Create comprehensive ring comparison visualization"""
    # Load data
    orig_ideas, orig_times = load_original_ring()
    opt_ideas, opt_times = load_optimized_ring()
    
    # Analyze both rings
    print("=" * 60)
    print("🔍 COMPREHENSIVE RING COMPARISON ANALYSIS")
    print("=" * 60)
    
    orig_closure, orig_avg, orig_transitions = analyze_closure_quality(orig_ideas, "ORIGINAL RING")
    opt_closure, opt_avg, opt_transitions = analyze_closure_quality(opt_ideas, "OPTIMIZED RING")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Mathematical Ring Optimization Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Ring closure comparison
    categories = ['Closure\\nSimilarity', 'Average\\nSimilarity', 'Transition\\nSmoothness']
    orig_scores = [orig_closure, orig_avg, np.mean(orig_transitions[:-1])]
    opt_scores = [opt_closure, opt_avg, np.mean(opt_transitions[:-1])]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, orig_scores, width, label='Original', color='lightcoral', alpha=0.7)
    axes[0, 0].bar(x + width/2, opt_scores, width, label='Optimized', color='lightblue', alpha=0.7)
    axes[0, 0].set_xlabel('Quality Metrics')
    axes[0, 0].set_ylabel('Similarity Score')
    axes[0, 0].set_title('Ring Quality Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(categories)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, opt) in enumerate(zip(orig_scores, opt_scores)):
        axes[0, 0].text(i - width/2, orig + 0.01, f'{orig:.3f}', ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, opt + 0.01, f'{opt:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Coverage comparison
    coverage_data = ['Original', 'Optimized']
    coverage_values = [len(orig_ideas), len(opt_ideas)]
    coverage_percentages = [(len(orig_ideas)/144)*100, (len(opt_ideas)/144)*100]
    
    bars = axes[0, 1].bar(coverage_data, coverage_values, color=['lightcoral', 'lightblue'], alpha=0.7)
    axes[0, 1].set_ylabel('Number of Time Slots')
    axes[0, 1].set_title('Ring Coverage (144 = Full 12 Hours)')
    axes[0, 1].axhline(y=144, color='red', linestyle='--', alpha=0.7, label='Full Coverage')
    
    # Add percentage labels
    for bar, pct in zip(bars, coverage_percentages):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{int(height)}\n({pct:.1f}%)', ha='center', va='bottom')
    
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Transition smoothness over time
    axes[1, 0].plot(range(len(orig_transitions)), orig_transitions, 'o-', 
                   label='Original', color='red', alpha=0.7, markersize=3)
    
    # Plot only matching length for comparison
    compare_length = min(len(orig_transitions), len(opt_transitions))
    axes[1, 0].plot(range(compare_length), opt_transitions[:compare_length], 'o-', 
                   label=f'Optimized (first {compare_length})', color='blue', alpha=0.7, markersize=3)
    
    # Highlight wrap-around transitions
    if len(orig_transitions) > 0:
        axes[1, 0].plot(len(orig_transitions)-1, orig_transitions[-1], 'ro', 
                       markersize=8, label='Original wrap-around')
    if len(opt_transitions) > 0:
        axes[1, 0].plot(len(opt_transitions)-1, opt_transitions[-1], 'bo', 
                       markersize=8, label='Optimized wrap-around')
    
    axes[1, 0].set_xlabel('Transition Index')
    axes[1, 0].set_ylabel('Similarity Score')
    axes[1, 0].set_title('Transition Smoothness Over Ring')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Circular ring representation
    theta_orig = np.linspace(0, 2*np.pi, len(orig_ideas), endpoint=False)
    theta_opt = np.linspace(0, 2*np.pi, len(opt_ideas), endpoint=False)
    
    # Original ring
    circle1 = patches.Circle((0, 0), 0.8, fill=False, color='red', linewidth=2, alpha=0.7)
    axes[1, 1].add_patch(circle1)
    
    # Optimized ring  
    circle2 = patches.Circle((0, 0), 1.0, fill=False, color='blue', linewidth=2, alpha=0.7)
    axes[1, 1].add_patch(circle2)
    
    # Mark start/end points
    axes[1, 1].plot(0.8*np.cos(theta_orig[0]), 0.8*np.sin(theta_orig[0]), 'ro', 
                   markersize=10, label='Original start/end')
    axes[1, 1].plot(1.0*np.cos(theta_opt[0]), 1.0*np.sin(theta_opt[0]), 'bo', 
                   markersize=10, label='Optimized start/end')
    
    # Draw closure lines
    axes[1, 1].plot([0.8*np.cos(theta_orig[0]), 0.8*np.cos(theta_orig[-1])],
                   [0.8*np.sin(theta_orig[0]), 0.8*np.sin(theta_orig[-1])],
                   'r--', alpha=0.7, linewidth=2, label=f'Original closure: {orig_closure:.3f}')
    axes[1, 1].plot([1.0*np.cos(theta_opt[0]), 1.0*np.cos(theta_opt[-1])],
                   [1.0*np.sin(theta_opt[0]), 1.0*np.sin(theta_opt[-1])],
                   'b--', alpha=0.7, linewidth=2, label=f'Optimized closure: {opt_closure:.3f}')
    
    axes[1, 1].set_xlim(-1.3, 1.3)
    axes[1, 1].set_ylim(-1.3, 1.3)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].set_title('Ring Closure Visualization')
    axes[1, 1].legend(loc='upper right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = "/tmp/IR/ring_comparison_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: {output_file}")
    
    plt.show()
    
    # Final comparison summary
    print(f"\n" + "="*60)
    print("🏆 FINAL MATHEMATICAL COMPARISON")
    print("="*60)
    
    improvements = []
    if opt_closure > orig_closure:
        improvements.append(f"✅ Closure: {((opt_closure - orig_closure)/abs(orig_closure)*100):+.1f}%")
    else:
        improvements.append(f"❌ Closure: {((opt_closure - orig_closure)/abs(orig_closure)*100):+.1f}%")
        
    if len(opt_ideas) > len(orig_ideas):
        improvements.append(f"✅ Coverage: +{len(opt_ideas) - len(orig_ideas)} slots")
    
    if np.mean(opt_transitions[:-1]) > np.mean(orig_transitions[:-1]):
        improvements.append(f"✅ Smoothness: {((np.mean(opt_transitions[:-1]) - np.mean(orig_transitions[:-1]))/np.mean(orig_transitions[:-1])*100):+.1f}%")
    else:
        improvements.append(f"❌ Smoothness: {((np.mean(opt_transitions[:-1]) - np.mean(orig_transitions[:-1]))/np.mean(orig_transitions[:-1])*100):+.1f}%")
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print(f"\n🎯 MATHEMATICAL VERDICT:")
    if opt_closure > orig_closure and len(opt_ideas) == 144:
        print("  🏆 OPTIMIZATION SUCCESSFUL - Better closure + full coverage!")
    elif len(opt_ideas) == 144:
        print("  ⚡ PARTIAL SUCCESS - Full coverage achieved, closure needs work")
    else:
        print("  🔄 NEEDS REFINEMENT - Try different optimization parameters")

if __name__ == "__main__":
    create_ring_visualization()