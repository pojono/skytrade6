import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks

def evaluate_fibonacci():
    # Load ratios
    ratios = np.load('fibonacci_ratios_expanded.npy')
    print(f"Total ratios loaded: {len(ratios)}")
    
    # Filter valid retracements (e.g. up to 1.5)
    ratios = ratios[(ratios > 0.05) & (ratios < 1.5)]
    print(f"Filtered ratios: {len(ratios)}")
    
    # Define standard Fibonacci levels
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272]
    
    # Plot histogram
    plt.figure(figsize=(15, 8))
    
    # Create histogram with high bin count to find natural clustering
    counts, bins, patches = plt.hist(ratios, bins=200, density=True, alpha=0.6, color='blue', label='Empirical Distribution')
    
    # Fit KDE
    kde = stats.gaussian_kde(ratios, bw_method=0.02)
    x = np.linspace(0.05, 1.5, 1000)
    kde_y = kde(x)
    plt.plot(x, kde_y, color='red', lw=2, label='KDE')
    
    # Find peaks in KDE
    peaks, _ = find_peaks(kde_y, prominence=0.05)
    peak_x = x[peaks]
    peak_y = kde_y[peaks]
    
    # Plot empirical peaks
    plt.scatter(peak_x, peak_y, color='black', s=50, zorder=5, label='Empirical Peaks')
    for px, py in zip(peak_x, peak_y):
        plt.annotate(f"{px:.3f}", (px, py), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    # Add vertical lines for Fibonacci levels
    for fib in fib_levels:
        plt.axvline(x=fib, color='green', linestyle='--', alpha=0.7, label=f'Fib {fib}' if fib == 0.236 else "")
        
    plt.title('Distribution of Retracement Ratios in Crypto Markets\n(20 Top Altcoins - 5m to 1d)')
    plt.xlabel('Retracement Ratio (relative to swing size)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('fibonacci_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Quantitative Analysis: Do peaks align with Fib levels better than chance?
    print("\n--- KDE Peaks ---")
    for px in peak_x:
        print(f"Peak at: {px:.4f}")
        
    print("\n--- Distance to nearest Fibonacci Level ---")
    distances = []
    for px in peak_x:
        nearest_fib = min(fib_levels, key=lambda f: abs(f - px))
        dist = abs(nearest_fib - px)
        distances.append(dist)
        print(f"Peak: {px:.4f} -> Nearest Fib: {nearest_fib} (Diff: {dist:.4f})")
        
    avg_dist = np.mean(distances)
    print(f"\nAverage distance from peaks to Fib levels: {avg_dist:.4f}")
    
    # Random chance baseline (Monte Carlo)
    print("\nSimulating random peaks to find baseline expected distance...")
    random_avg_dists = []
    for _ in range(10000):
        # Generate random peaks in [0.05, 1.5]
        random_peaks = np.random.uniform(0.05, 1.5, len(peak_x))
        dists = [min(abs(f - rp) for f in fib_levels) for rp in random_peaks]
        random_avg_dists.append(np.mean(dists))
        
    p_value = np.sum(np.array(random_avg_dists) <= avg_dist) / 10000
    print(f"Expected random distance: {np.mean(random_avg_dists):.4f}")
    print(f"P-value (probability of this alignment by chance): {p_value:.4f}")
    
    if p_value < 0.05:
        print("CONCLUSION: Statistically significant alignment with Fibonacci levels.")
    else:
        print("CONCLUSION: Reject Golden Ratio/Fibonacci theory. The peaks in market structure do not align with Fib levels better than random chance.")

if __name__ == "__main__":
    evaluate_fibonacci()
