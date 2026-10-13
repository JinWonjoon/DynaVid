import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import pdb

def normalize_counts(counts: torch.Tensor, bin_width: float):
    total = counts.sum()
    if total > 0:
        # PDF 정규화: count / (total * bin_width)
        return (counts / (total * bin_width) * 10).numpy()
    else:
        return counts.numpy()


def plot_histogram_three_methods(
    counts_ours,
    counts_vj,
    counts_flovd,
    bin_edges,
    save_path="flow_histogram_results/flow_RGB_magnitude_histogram.png",
    save_data=True,
):
    """
    counts_*: 1D CPU tensor (bin counts)
    bin_edges: 1D CPU tensor (len = bins + 1)
    save_path: 최종 저장 파일 경로 (.png 포함)
    save_data: True이면 .npz로 histogram 데이터 저장
    """
    save_dir = os.path.dirname(save_path)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    bins = len(counts_ours)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width = (bin_edges[1] - bin_edges[0]).item()
    
    bin_width_total = (bin_edges[-1] - bin_edges[0]).item()

    # Normalize (확률분포)
    pdf_ours = normalize_counts(counts_ours, bin_width_total)
    pdf_vj = normalize_counts(counts_vj, bin_width_total)
    pdf_flovd = normalize_counts(counts_flovd, bin_width_total)

    # -------------------------------------------------------
    # 1️⃣ Combined histogram
    # -------------------------------------------------------
    plt.figure(figsize=(4, 4))
    plt.bar(bin_centers.numpy(), pdf_ours, width=bin_width, alpha=0.4, label="Ours")
    plt.bar(bin_centers.numpy(), pdf_vj, width=bin_width, alpha=0.4, label="VideoJam")
    # plt.bar(bin_centers.numpy(), pdf_flovd, width=bin_width, alpha=0.4, label="FloVD")

    plt.xlabel("Magnitude")
    plt.ylabel("Density")
    plt.title("Normalized Flow Magnitude Distribution")
    plt.legend(fontsize = 12)
    plt.tight_layout()

    combined_path = save_path
    plt.savefig(combined_path, dpi=300)
    plt.close()
    print(f"[✓] Saved combined histogram → {combined_path}")

    # -------------------------------------------------------
    # 2️⃣ Individual histograms
    # -------------------------------------------------------
    base = os.path.splitext(save_path)[0]

    def save_single(pdf, label, color, suffix):
        plt.figure(figsize=(6, 4))
        plt.bar(bin_centers.numpy(), pdf, width=bin_width, color=color, alpha=0.6, label=label)
        plt.xlabel("Magnitude")
        plt.ylabel("Density")
        plt.title(f"Normalized Flow Magnitude Distribution")
        plt.legend(fontsize = 12)
        plt.tight_layout()
        out_path = f"{base}_{suffix}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[✓] Saved {label} histogram → {out_path}")

    save_single(pdf_ours, "Ours", "C0", "ours")
    save_single(pdf_vj, "VideoJam", "C1", "vj")
    # save_single(pdf_flovd, "FloVD", "C2", "flovd")

    # -------------------------------------------------------
    # 3️⃣ Histogram raw data 저장 (.npz)
    # -------------------------------------------------------
    if save_data:
        data_path = f"{base}_data.npz"
        np.savez(
            data_path,
            bin_edges=bin_edges.numpy(),
            counts_ours=counts_ours.numpy(),
            counts_vj=counts_vj.numpy(),
            counts_flovd=counts_flovd.numpy(),
        )
        print(f"[✓] Saved histogram data → {data_path}")
        


data = np.load("flow_histogram_results_120samples/flow_RGB_magnitude_histogram_data.npz")

NUM=666

counts_ours = torch.from_numpy(data["counts_ours"])[:NUM]
counts_vj = torch.from_numpy(data["counts_vj"])[:NUM]
counts_flovd = torch.from_numpy(data["counts_flovd"])[:NUM]
bin_edges = torch.from_numpy(data["bin_edges"])[:NUM+1]


plot_histogram_three_methods(
    counts_ours,
    counts_vj,
    counts_flovd,
    bin_edges,
    save_path="flow_histogram_results_120samples/flow_RGB_magnitude_histogram_replot_Ours_VJ.png",
    save_data=False,  # 이미 저장된 데이터로부터 그릴 때는 False
)