#!/usr/bin/env python
"""
Ground-truth evaluator for Argus object detection, tracking, and evolutionary optimization.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two [x, y, w, h] boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class GroundTruthEvaluator:
    def __init__(self, ground_truth_file: Path, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        with open(ground_truth_file, "r") as f:
            self.gt_data = json.load(f)

    def evaluate_predictions(self, predictions_by_frame: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate predicted frame detections against loaded ground truth.
        """
        tp = 0
        fp = 0
        fn = 0
        id_switches = 0
        gt_track_to_pred_track: Dict[int, int] = {}

        frames = self.gt_data.get("frames", [])
        total_gt_objects = 0

        for frame in frames:
            fid = frame["frame_id"]
            gt_objects = frame.get("annotations", [])
            pred_objects = predictions_by_frame.get(fid, [])
            total_gt_objects += len(gt_objects)

            matched_preds = set()
            for gt in gt_objects:
                gt_box = gt["bbox"]
                gt_track = gt.get("track_id")
                best_iou = 0.0
                best_pred_idx = -1

                for idx, pred in enumerate(pred_objects):
                    if idx in matched_preds:
                        continue
                    if pred.get("class") != gt.get("class"):
                        continue
                    iou = compute_iou(gt_box, pred["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred_idx = idx

                if best_iou >= self.iou_threshold and best_pred_idx >= 0:
                    tp += 1
                    matched_preds.add(best_pred_idx)
                    matched_pred = pred_objects[best_pred_idx]
                    pred_track = matched_pred.get("track_id")

                    if gt_track is not None and pred_track is not None:
                        if gt_track in gt_track_to_pred_track:
                            if gt_track_to_pred_track[gt_track] != pred_track:
                                id_switches += 1
                                gt_track_to_pred_track[gt_track] = pred_track
                        else:
                            gt_track_to_pred_track[gt_track] = pred_track
                else:
                    fn += 1

            fp += len(pred_objects) - len(matched_preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / len(frames) if frames else 0.0

        return {
            "total_frames_evaluated": len(frames),
            "ground_truth_objects": total_gt_objects,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate_per_frame": round(fpr, 4),
            "id_switches": id_switches,
        }


def main():
    p = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    p.add_argument("--ground-truth", default="data/evaluation/v1/annotations/ground_truth.json", type=Path)
    p.add_argument("--iou-threshold", default=0.5, type=float)
    args = p.parse_args()

    evaluator = GroundTruthEvaluator(args.ground_truth, iou_threshold=args.iou_threshold)

    # Self-test using exact GT as mock predictions
    mock_preds = {}
    for f in evaluator.gt_data.get("frames", []):
        mock_preds[f["frame_id"]] = f.get("annotations", [])

    results = evaluator.evaluate_predictions(mock_preds)
    print("\n" + "=" * 60)
    print("ARGUS EVALUATION HARNESS RESULTS:")
    print("=" * 60)
    for k, v in results.items():
        print(f"  {k:<32}: {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
