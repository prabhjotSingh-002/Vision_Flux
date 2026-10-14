#!/usr/bin/env python3
"""
PixelForge Benchmark Plotter
Reads benchmark CSV results and generates comparative performance charts.
"""

import sys
import os
import csv

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("benchmarks", "results.csv")
    output_img = sys.argv[2] if len(sys.argv) > 2 else os.path.join("benchmarks", "benchmark_plot.png")

    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        print("Run benchmarks first using: ./build/pixelforge benchmark")
        sys.exit(1)

    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "width": int(row["input_width"]),
                    "height": int(row["input_height"]),
                    "scale": int(row["scale"]),
                    "cpu_compute_ms": float(row["cpu_compute_ms"]),
                    "gpu_kernel_ms": float(row["gpu_kernel_ms"]),
                    "gpu_end_to_end_ms": float(row["gpu_end_to_end_ms"]),
                    "speedup": float(row["speedup"]),
                })
            except (KeyError, ValueError) as e:
                print(f"Skipping malformed row: {row} ({e})")

    if not rows:
        print("[Error] No valid data rows found in CSV.")
        sys.exit(1)

    print(f"Loaded {len(rows)} benchmark entries from {csv_path}:")
    print(f"{'Input Size':<15} {'Scale':<8} {'CPU (ms)':<12} {'GPU Kernel':<12} {'GPU E2E':<12} {'Speedup':<10}")
    print("-" * 70)
    for r in rows:
        size_str = f"{r['width']}x{r['height']}"
        print(f"{size_str:<15} {r['scale']}x      {r['cpu_compute_ms']:<12.3f} {r['gpu_kernel_ms']:<12.3f} {r['gpu_end_to_end_ms']:<12.3f} {r['speedup']:<10.2f}x")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [f"{r['width']}x{r['height']}\n({r['scale']}x)" for r in rows]
        x = np.arange(len(labels))
        width = 0.25

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot 1: Processing Times (Log Scale)
        cpu_vals = [r["cpu_compute_ms"] for r in rows]
        gpu_k_vals = [r["gpu_kernel_ms"] for r in rows]
        gpu_e2e_vals = [r["gpu_end_to_end_ms"] for r in rows]

        ax1.bar(x - width, cpu_vals, width, label="CPU Reference", color="#ef4444")
        ax1.bar(x, gpu_e2e_vals, width, label="GPU End-to-End", color="#3b82f6")
        ax1.bar(x + width, gpu_k_vals, width, label="GPU Kernel Only", color="#10b981")

        ax1.set_ylabel("Execution Time (ms, log scale)")
        ax1.set_title("PixelForge: CPU vs GPU Execution Time")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_yscale("log")
        ax1.grid(True, linestyle="--", alpha=0.5, which="both")
        ax1.legend()

        # Subplot 2: Speedup (CPU / GPU E2E)
        speedup_vals = [r["speedup"] for r in rows]
        bars = ax2.bar(x, speedup_vals, width=0.45, color="#8b5cf6")
        ax2.axhline(1.0, color="gray", linestyle="--", label="1x Baseline")
        ax2.set_ylabel("Speedup Factor (x)")
        ax2.set_title("PixelForge: GPU End-to-End Speedup over CPU")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}x",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        plt.savefig(output_img, dpi=300)
        print(f"\n[Success] Benchmark plot saved to: {output_img}")

    except ImportError:
        print("\n[Notice] matplotlib is not installed. To generate the chart, run: pip install matplotlib")

if __name__ == "__main__":
    main()
