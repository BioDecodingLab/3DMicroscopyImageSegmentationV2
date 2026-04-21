import concurrent.futures
from pathlib import Path
from typing import Dict, List

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import tifffile

from src.config import AVAILABLE_METRICS, MAX_WORKERS, OUTPUT_EXTENSION
from src.inference.utils import apply_threshold_to_image_and_convert_to_dtype

# Image-level metrics operate on full reconstructed volumes (potentially large);
# keep the pool small to bound peak memory.
IMAGE_LEVEL_MAX_WORKERS = 1


def compute_metrics_with_true_and_pred(
    image_true: np.ndarray, image_pred: np.ndarray, metrics_to_compute: List[str]
) -> Dict[str, float]:
    """Compute segmentation metrics between ground truth and prediction."""

    image_true = apply_threshold_to_image_and_convert_to_dtype(image_true, 0, int)
    image_pred = apply_threshold_to_image_and_convert_to_dtype(image_pred, 0, int)

    y_true = image_true.flatten()
    y_pred = image_pred.flatten()

    # Always compute the confusion matrix so downstream metrics don't depend on
    # a predicate that also happens to gate binding of tp/fp/tn/fn.
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    metrics: Dict[str, float] = {}

    if "precision" in metrics_to_compute:
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)

    if "recall" in metrics_to_compute:
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)

    if "f1" in metrics_to_compute:
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    if "iou" in metrics_to_compute:
        metrics["iou"] = jaccard_score(y_true, y_pred, zero_division=0)

    if "accuracy" in metrics_to_compute:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if "specificity" in metrics_to_compute:
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    if "sensitivity" in metrics_to_compute:
        if "recall" in metrics:
            metrics["sensitivity"] = metrics["recall"]
        else:
            metrics["sensitivity"] = recall_score(y_true, y_pred, zero_division=0)

    if "dice" in metrics_to_compute:
        metrics["dice"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

    if "volume_similarity" in metrics_to_compute:
        metrics["volume_similarity"] = (
            1 - abs((fn - fp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0
        )

    return metrics


def read_and_compute_metrics(
    image_true_path: Path, image_pred_path: Path, metrics_to_compute: List[str]
):
    image_true = tifffile.imread(str(image_true_path))
    image_pred = tifffile.imread(str(image_pred_path))

    return compute_metrics_with_true_and_pred(image_true, image_pred, metrics_to_compute)


def results_to_dataframe(results):
    return pd.DataFrame(results)


def plot_metrics_boxplots(df, save_path=None):
    available_metrics = [m for m in df.columns if m != "method"]

    n_metrics = len(available_metrics)
    if n_metrics == 0:
        return

    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        df.boxplot(column=metric, by="method", ax=ax)
        ax.set_title(f"{metric.replace('_', ' ').title()} Distribution")
        ax.set_xlabel("method")
        ax.set_ylabel(metric)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def _collect_metrics_for_predictions(
    predictions_dir: Path,
    get_ground_truth_path,
    max_workers: int,
) -> list:
    """For each method directory under predictions_dir, compute metrics for every
    prediction file paired with its ground truth via get_ground_truth_path.

    Files without a matching ground truth are skipped with a warning rather than
    crashing the whole plotting job.
    """
    results = []
    for method in sorted(predictions_dir.glob("*")):
        if not method.is_dir():
            continue

        pairs = []
        for pred_path in method.glob("*"):
            gt_path = get_ground_truth_path(pred_path)
            if gt_path is None or not gt_path.is_file():
                logger.warning(f"No ground truth for {pred_path}, skipping")
                continue
            pairs.append((pred_path, gt_path))

        if not pairs:
            continue

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    read_and_compute_metrics, pred_path, gt_path, AVAILABLE_METRICS
                ): pred_path
                for pred_path, gt_path in pairs
            }
            for future in concurrent.futures.as_completed(futures):
                pred_path = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Failed computing metrics for {pred_path}: {e}")
                    continue
                result["method"] = method.stem
                results.append(result)

    return results


def run_plotting(
    predictions_patch_level: Path,
    predictions_image_level: Path,
    ground_truth_patches_dir: Path,
    ground_truth_masks_dir: Path,
    output_dir: Path,
) -> None:
    """Generate evaluation plots for all prediction methods.

    Args:
        predictions_patch_level: Directory with patch-level predictions
        predictions_image_level: Directory with image-level predictions
        ground_truth_patches_dir: Directory with ground truth mask patches
            (expected to be the no-padding regular patches)
        ground_truth_masks_dir: Directory with complete ground truth masks
        output_dir: Directory to save plot figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    patches_dict = {
        p.stem: p for p in ground_truth_patches_dir.rglob(f"*{OUTPUT_EXTENSION}") if p.is_file()
    }

    logger.info("Computing patch-level metrics...")
    patch_results = _collect_metrics_for_predictions(
        predictions_patch_level,
        get_ground_truth_path=lambda pred_path: patches_dict.get(pred_path.stem),
        max_workers=MAX_WORKERS,
    )

    if patch_results:
        df = results_to_dataframe(patch_results)
        plot_metrics_boxplots(df, output_dir / "metrics_patches.png")
        logger.info(f"Saved patch-level plot to {output_dir / 'metrics_patches.png'}")

    logger.info("Computing image-level metrics...")
    image_results = _collect_metrics_for_predictions(
        predictions_image_level,
        get_ground_truth_path=lambda pred_path: ground_truth_masks_dir / pred_path.name,
        max_workers=IMAGE_LEVEL_MAX_WORKERS,
    )

    if image_results:
        df = results_to_dataframe(image_results)
        plot_metrics_boxplots(df, output_dir / "metrics_images.png")
        logger.info(f"Saved image-level plot to {output_dir / 'metrics_images.png'}")
