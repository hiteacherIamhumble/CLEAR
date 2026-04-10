from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class BEVPostprocessResult:
    """Container for position-band post-process selection."""

    keep_indices: torch.Tensor
    keypoint_scores: np.ndarray
    stage_counts: dict[str, int] | None = None
    stage_counts_eval: dict[str, int] | None = None
    stage_thresholds: dict[str, float] | None = None


class PositionBandPostProcessor:
    """Per-y-band calibrated BEV post-processing.

    Runtime behavior is intentionally simple:
      1) map the predicted player position keypoint to a vertical image band,
      2) read the learned threshold ratio assigned to that band,
      3) apply ``base_score_threshold * threshold_ratio`` to bbox confidence,
      4) optionally require a minimum keypoint score only for detections that
         were rescued by the relaxed local threshold.

    This keeps the original inference logic intact for high-confidence
    detections while allowing validation-calibrated bands to become stricter or
    more relaxed than the global threshold.
    """

    def __init__(self, stats: dict[str, Any]) -> None:
        self.stats = stats

        self.position_kpt_index = int(stats.get("position_keypoint_index", 1))
        self.rescue_keypoint_ratio = max(float(stats.get("rescue_keypoint_ratio", 0.9)), 0.0)
        self.base_score_threshold = float(stats.get("base_score_threshold", stats.get("global_score_threshold", 0.0)))

        image_bounds = stats.get("image_position_bounds") or {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
        self.pos_y_min = float(image_bounds["y_min"])
        self.pos_y_max = float(image_bounds["y_max"])

        banding = stats.get("banding", {})
        self.y_bands = max(int(banding.get("y_bands", stats.get("y_bands", 4))), 1)

        self.fallback_threshold_ratio = float(stats.get("fallback_threshold_ratio", 1.0))
        self.band_ratio_lookup = {
            key: float(value.get("threshold_ratio", self.fallback_threshold_ratio))
            for key, value in stats.get("band_threshold_lookup", {}).items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> PositionBandPostProcessor:
        with Path(path).open() as f:
            payload = json.load(f)
        return cls(payload)

    def set_base_score_threshold(self, score_threshold: float) -> None:
        self.base_score_threshold = float(score_threshold)

    def apply(
        self,
        predn_scaled: dict[str, torch.Tensor],
        ori_shape: tuple[int, int],
    ) -> BEVPostprocessResult:
        """Apply y-band thresholds and return selected indices."""
        n = predn_scaled["conf"].shape[0]
        if n == 0:
            empty = torch.empty((0,), dtype=torch.long, device=predn_scaled["conf"].device)
            return BEVPostprocessResult(
                empty,
                np.zeros((0,), dtype=np.float32),
                stage_counts={"input": 0, "band_conf": 0, "rescue_kpt": 0, "final": 0},
                stage_counts_eval={"input": 0, "band_conf": 0, "rescue_kpt": 0, "final": 0},
                stage_thresholds={
                    "base_score_threshold": float(self.base_score_threshold),
                    "band_low_ratio": float(min(self.band_ratio_lookup.values(), default=self.fallback_threshold_ratio)),
                    "band_high_ratio": float(max(self.band_ratio_lookup.values(), default=self.fallback_threshold_ratio)),
                    "num_bands": float(self.y_bands),
                    "fallback_threshold_ratio": float(self.fallback_threshold_ratio),
                    "rescue_keypoint_ratio": float(self.rescue_keypoint_ratio),
                    "local_threshold_min": 0.0,
                    "local_threshold_max": 0.0,
                },
            )

        kpts = predn_scaled["kpts"].detach().cpu().numpy()
        keypoint_scores = self._keypoint_scores(predn_scaled)
        conf_scores = predn_scaled["conf"].detach().cpu().numpy().astype(np.float32)
        pos_y = self._normalized_image_positions_y(kpts, ori_shape)
        valid_pos = np.isfinite(pos_y)
        band_indices = np.full(n, -1, dtype=np.int32)
        for i in np.where(valid_pos)[0]:
            band_indices[i] = self._band_index(float(pos_y[i]))

        local_ratios = np.full(n, self.fallback_threshold_ratio, dtype=np.float32)
        if self.base_score_threshold > 0:
            for i in np.where(valid_pos)[0]:
                local_ratios[i] = self._ratio_for_position_y(pos_y[i])
        local_thresholds = np.maximum(self.base_score_threshold * local_ratios, 0.0)
        keep = conf_scores >= local_thresholds
        band_conf_mask = keep.copy()
        band_conf_count = int(np.count_nonzero(keep))

        rescued = keep & (conf_scores < self.base_score_threshold)
        if np.any(rescued) and self.rescue_keypoint_ratio > 0:
            rescue_threshold = float(self.base_score_threshold) * float(self.rescue_keypoint_ratio)
            keep[rescued] &= keypoint_scores[rescued] >= rescue_threshold
        rescue_kpt_mask = keep.copy()
        rescue_kpt_count = int(np.count_nonzero(keep))

        eval_mask = conf_scores >= self.base_score_threshold if self.base_score_threshold > 0 else np.ones(n, dtype=bool)
        eval_input = int(np.count_nonzero(eval_mask))
        eval_band_conf = int(np.count_nonzero(band_conf_mask & eval_mask))
        eval_rescue_kpt = int(np.count_nonzero(rescue_kpt_mask & eval_mask))

        kept_indices = np.where(keep)[0]
        keep_tensor = torch.as_tensor(kept_indices, dtype=torch.long, device=predn_scaled["conf"].device)
        return BEVPostprocessResult(
            keep_tensor,
            keypoint_scores,
            stage_counts={
                "input": int(n),
                "band_conf": band_conf_count,
                "rescue_kpt": rescue_kpt_count,
                "final": int(len(kept_indices)),
            },
            stage_counts_eval={
                "input": eval_input,
                "band_conf": eval_band_conf,
                "rescue_kpt": eval_rescue_kpt,
                "final": eval_rescue_kpt,
            },
            stage_thresholds={
                "base_score_threshold": float(self.base_score_threshold),
                "band_low_ratio": float(min(self.band_ratio_lookup.values(), default=self.fallback_threshold_ratio)),
                "band_high_ratio": float(max(self.band_ratio_lookup.values(), default=self.fallback_threshold_ratio)),
                "num_bands": float(self.y_bands),
                "fallback_threshold_ratio": float(self.fallback_threshold_ratio),
                "rescue_keypoint_ratio": float(self.rescue_keypoint_ratio),
                "local_threshold_min": float(local_thresholds.min()) if local_thresholds.size else 0.0,
                "local_threshold_max": float(local_thresholds.max()) if local_thresholds.size else 0.0,
            },
        )

    def _keypoint_scores(self, predn_scaled: dict[str, torch.Tensor]) -> np.ndarray:
        kpts = predn_scaled["kpts"].detach().cpu().numpy()
        if kpts.ndim == 3 and kpts.shape[1] > self.position_kpt_index and kpts.shape[2] >= 3:
            return kpts[:, self.position_kpt_index, 2].astype(np.float32)
        return predn_scaled["conf"].detach().cpu().numpy().astype(np.float32)

    def _normalized_image_positions_y(self, kpts: np.ndarray, ori_shape: tuple[int, int]) -> np.ndarray:
        out = np.full((kpts.shape[0],), np.nan, dtype=np.float32)
        if kpts.ndim != 3 or kpts.shape[1] <= self.position_kpt_index:
            return out
        h = float(ori_shape[0])
        if h <= 0:
            return out
        pts = kpts[:, self.position_kpt_index, :2]
        out[:] = pts[:, 1] / h
        return out

    def _band_index(self, y: float) -> int:
        y_span = max(self.pos_y_max - self.pos_y_min, 1e-6)
        by = int(np.floor((y - self.pos_y_min) / y_span * self.y_bands))
        return min(max(by, 0), self.y_bands - 1)

    def _ratio_for_position_y(self, y: float) -> float:
        key = str(self._band_index(y))
        return float(self.band_ratio_lookup.get(key, self.fallback_threshold_ratio))
