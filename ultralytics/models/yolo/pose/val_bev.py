# Ultralytics BEV Pose Validator with sskit LocSim evaluation support.
# Mirrors the mmpose CocoMetric locsim/locsim_bbox evaluation used in:
#   mmpose/mmpose/evaluation/metrics/coco_metric.py
#   mmpose/configs/body_bev_position/spiideo_soccernet/synloc.py
#
# Key evaluation behaviour reproduced from mmpose:
#   - Runs standard val first to find score_threshold (saved as *_val_stats.json)
#   - Uses sskit.coco.LocSimCOCOeval for LocSim mAP (position_from_keypoint_index=1)
#   - Uses sskit.coco.BBoxLocSimCOCOeval for BBox LocSim mAP
#   - Applies the val score_threshold when evaluating on test / challenge splits
#   - Writes results.json + metadata.json and zips for challenge submission

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from xtcocotools.coco import COCO

from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA

from .postprocess_bev import PositionBandPostProcessor
from .val import PoseValidator


class BEVPoseValidator(PoseValidator):
    """Extends PoseValidator with sskit LocSim evaluation.

    Mirrors the mmpose CocoMetric with iou_type='locsim'/'locsim_bbox'.

    Usage (programmatic):
        from ultralytics import YOLO
        model = YOLO("runs/pose-bev/yolo11m-pose-bev-640/weights/best.pt")
        validator = BEVPoseValidator(args=dict(
            data="soccernet-synloc.yaml",
            split="val",
            save_json=True,
        ))
        validator(model=model.model)

    Usage (via test_bev.py script):
        python test_bev.py --weights best.pt --data soccernet-synloc.yaml --split val
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        phase="val",
        bev_postprocess_stats: str | Path | None = None,
    ):
        """Initialize BEVPoseValidator.

        Args:
            phase: Evaluation phase — 'val', 'test', or 'challenge'.  Kept out
                   of the ``args`` dict so ultralytics get_cfg() does not reject
                   it as an unknown key.
        """
        # Store phase BEFORE super().__init__ so eval_json can see it even if
        # called during the parent's initialisation chain.
        self.phase = phase
        self.bev_postprocess_stats = Path(bev_postprocess_stats) if bev_postprocess_stats else None
        self.bev_postprocessor: PositionBandPostProcessor | None = None
        self._pp_stage_totals: dict[str, int] = {"input": 0, "band_conf": 0, "rescue_kpt": 0, "final": 0}
        self._pp_stage_eval_totals: dict[str, int] = {"input": 0, "band_conf": 0, "rescue_kpt": 0, "final": 0}
        self._pp_num_images = 0
        self._pp_last_thresholds: dict[str, float] = {}
        self._analysis_score_threshold: float | None = None
        self._final_prediction_threshold: float | None = None
        self._analysis_score_input = 0
        self._analysis_score_keep = 0
        super().__init__(dataloader, save_dir, args, _callbacks)

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize metrics, reading custom sigmas from dataset config."""
        super().init_metrics(model)
        # Ensure save_json is on so predictions.json is written for LocSim eval
        self.args.save_json = True
        self._init_bev_postprocessor()
        self._analysis_score_threshold = self._resolve_eval_score_threshold_for_analysis()
        self._final_prediction_threshold = self._resolve_final_prediction_threshold()

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with optional BEV y-band post-processing."""
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            need_scaled = self.args.save_json or self.args.save_txt or (self.bev_postprocessor is not None)
            predn_scaled = self.scale_preds(predn, pbatch) if need_scaled else None

            self._accumulate_score_threshold_debug(predn)

            if self.bev_postprocessor is not None and predn["cls"].shape[0] > 0 and predn_scaled is not None:
                pp_res = self.bev_postprocessor.apply(predn_scaled, pbatch["ori_shape"])
                self._accumulate_postprocess_debug(pp_res)
                predn = self._index_pred(predn, pp_res.keep_indices)
                predn_scaled = self._index_pred(predn_scaled, pp_res.keep_indices)
            elif (
                self._final_prediction_threshold is not None
                and predn["cls"].shape[0] > 0
                and predn_scaled is not None
            ):
                keep = torch.nonzero(predn["conf"] >= self._final_prediction_threshold, as_tuple=False).squeeze(-1)
                predn = self._index_pred(predn, keep)
                predn_scaled = self._index_pred(predn_scaled, keep)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            if self.args.save_json and predn_scaled is not None:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt and predn_scaled is not None:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Run LocSim evaluation via sskit after standard validation finishes.

        Mirrors mmpose CocoMetric._do_python_keypoint_eval() with
        iou_type='locsim' and iou_type='locsim_bbox'.

        Args:
            stats: Existing stats dict from YOLO validation loop.

        Returns:
            Updated stats dict with LocSim AP/AR metrics added.
        """
        self._log_score_threshold_summary()
        self._log_postprocess_summary()
        try:
            from sskit.coco import BBoxLocSimCOCOeval, LocSimCOCOeval
        except ImportError:
            LOGGER.warning(
                "sskit not installed — skipping LocSim evaluation. "
                "Install with: pip install sskit"
            )
            return stats

        pred_json = self.save_dir / "predictions.json"
        anno_json = self._get_anno_json()

        if not pred_json.exists():
            LOGGER.warning(f"predictions.json not found at {pred_json}; skipping LocSim eval.")
            return stats

        if not anno_json.exists():
            LOGGER.warning(f"Annotation file not found at {anno_json}; skipping LocSim eval.")
            return stats

        sigmas = list(self.sigma)  # [0.089, 0.089] for BEV task

        LOGGER.info(f"\nRunning sskit LocSim evaluation (phase='{self.phase}')...")

        # ---- build predictions in sskit-compatible COCO format ----
        coco_gt = COCO(str(anno_json))
        coco_dt = coco_gt.loadRes(str(pred_json))

        # ---- BBox LocSim ----
        stats = self._run_locsim(
            stats, coco_gt, coco_dt, sigmas,
            iou_type="locsim_bbox",
            EvalClass=BBoxLocSimCOCOeval,
            prefix="locsim_bbox",
        )

        # ---- Keypoint LocSim (primary metric) ----
        stats = self._run_locsim(
            stats, coco_gt, coco_dt, sigmas,
            iou_type="locsim",
            EvalClass=LocSimCOCOeval,
            prefix="locsim",
        )

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_anno_json(self) -> Path:
        """Return annotation JSON path for the current split."""
        data_path = Path(self.data["path"])
        split = getattr(self.args, "split", "val")
        # Try challenge annotation file first
        if split == "challenge":
            candidates = [
                data_path / "annotations" / "challenge_public.json",
                data_path / "annotations" / "challenge.json",
            ]
        elif split == "test":
            candidates = [data_path / "annotations" / "test.json"]
        else:
            candidates = [data_path / "annotations" / "val.json"]

        for c in candidates:
            if c.exists():
                return c

        # Final fallback: whatever the dataset YAML says
        return data_path / "annotations" / f"{split}.json"

    def _init_bev_postprocessor(self) -> None:
        """Initialize BEV postprocessor from stats file when configured."""
        if not self.bev_postprocess_stats:
            return
        if not self.bev_postprocess_stats.exists():
            LOGGER.warning(
                f"BEV postprocess stats file not found: {self.bev_postprocess_stats}. "
                "BEV y-band post-processing is disabled."
            )
            return
        try:
            self.bev_postprocessor = PositionBandPostProcessor.from_file(self.bev_postprocess_stats)
        except Exception as e:
            LOGGER.warning(f"Failed to load BEV postprocess stats: {e}. BEV y-band post-processing is disabled.")
            self.bev_postprocessor = None
            return

        self.bev_postprocessor.set_base_score_threshold(self._resolve_base_score_threshold())
        LOGGER.info(f"Enabled BEV y-band calibrated post-processing from {self.bev_postprocess_stats}")

    def _resolve_base_score_threshold(self) -> float:
        """Resolve score threshold baseline used by Stage 1."""
        val_stats_file = self.save_dir / "locsim_val_stats.json"
        resolved: float | None = None
        if val_stats_file.exists():
            try:
                with open(val_stats_file) as f:
                    resolved = float(json.load(f)["stats"]["score_threshold"])
            except Exception as e:
                LOGGER.warning(f"Failed reading score_threshold from {val_stats_file}: {e}")
        if (
            (resolved is None or resolved < 0.05)
            and self.bev_postprocessor is not None
            and self.bev_postprocessor.base_score_threshold > 0
        ):
            resolved = float(self.bev_postprocessor.base_score_threshold)
        if resolved is None:
            resolved = float(getattr(self.args, "conf", 0.001))
        if resolved < 0.05:
            LOGGER.warning(
                f"Resolved base score_threshold={resolved:.4f} is very low. "
                "If you want the original (non-postprocess) val threshold, run val first without "
                "--bev-postprocess-stats or set base_score_threshold in stats JSON."
            )
        return resolved

    def _resolve_eval_score_threshold_for_analysis(self) -> float | None:
        """Resolve evaluation score threshold for debug-only retention logging."""
        val_stats_file = self.save_dir / "locsim_val_stats.json"
        if val_stats_file.exists():
            try:
                with open(val_stats_file) as f:
                    return float(json.load(f)["stats"]["score_threshold"])
            except Exception as e:
                LOGGER.warning(f"Failed reading debug score_threshold from {val_stats_file}: {e}")
        if self.bev_postprocessor is not None and self.bev_postprocessor.base_score_threshold > 0:
            return float(self.bev_postprocessor.base_score_threshold)
        return None

    def _resolve_final_prediction_threshold(self) -> float | None:
        """Resolve threshold used to prefilter exported predictions.

        For baseline test/challenge runs, we want predictions.json/results.json to
        contain the same detections that participate in LocSim evaluation.
        """
        if self.phase not in ("test", "challenge"):
            return None
        if self.bev_postprocessor is not None:
            return None
        return self._analysis_score_threshold

    def _resolve_evaluator_threshold_for_filtered_predictions(self) -> float:
        """Return a reporting threshold that preserves the filtered prediction set.

        The evaluator still expects one scalar ``score_threshold`` for its
        thresholded point metrics (precision/recall/f1). We choose a threshold
        that does not drop any prediction already kept by runtime filtering:
          - baseline: the same global threshold used to filter predictions
          - y-band mode: the minimum threshold configured by postprocess
        """
        if self.bev_postprocessor is not None:
            if self._pp_last_thresholds:
                return float(self._pp_last_thresholds.get("local_threshold_min", 0.0))
            return 0.0
        if self._final_prediction_threshold is not None:
            return float(self._final_prediction_threshold)
        return 0.0

    @staticmethod
    def _index_pred(predn: dict[str, torch.Tensor], keep: torch.Tensor) -> dict[str, torch.Tensor]:
        """Index all prediction tensors by selected detection indices."""
        return {k: v[keep] for k, v in predn.items()}

    def _accumulate_postprocess_debug(self, pp_res) -> None:
        """Accumulate stage-wise retention counters across the dataset."""
        if not pp_res.stage_counts:
            return
        self._pp_num_images += 1
        for key in self._pp_stage_totals:
            self._pp_stage_totals[key] += int(pp_res.stage_counts.get(key, 0))
        if pp_res.stage_counts_eval:
            for key in self._pp_stage_eval_totals:
                self._pp_stage_eval_totals[key] += int(pp_res.stage_counts_eval.get(key, 0))
        if pp_res.stage_thresholds:
            self._pp_last_thresholds = dict(pp_res.stage_thresholds)

    def _accumulate_score_threshold_debug(self, predn: dict[str, torch.Tensor]) -> None:
        """Accumulate retention stats for final score-threshold filtering."""
        if self._analysis_score_threshold is None:
            return
        conf = predn["conf"]
        if conf.numel() == 0:
            return
        conf_np = conf.detach().cpu().numpy()
        self._analysis_score_input += int(conf_np.size)
        self._analysis_score_keep += int(np.count_nonzero(conf_np >= self._analysis_score_threshold))

    def _log_score_threshold_summary(self) -> None:
        """Log retention if only score-threshold filtering is applied."""
        if self._analysis_score_threshold is None or self._analysis_score_input == 0:
            return
        inp = self._analysis_score_input
        keep = self._analysis_score_keep
        LOGGER.info(
            "Score-threshold retention before BEV postprocess: "
            f"input={inp}, keep={keep} ({keep / max(inp, 1):.3f}), "
            f"threshold={self._analysis_score_threshold:.4f}"
        )

    def _log_postprocess_summary(self) -> None:
        """Log stage-wise post-process retention summary."""
        if self.bev_postprocessor is None or self._pp_num_images == 0:
            return
        totals = self._pp_stage_totals
        n_in = max(totals["input"], 1)
        msg = (
            "BEV calibrated-band post-process retention (all detections): "
            f"input={totals['input']}, "
            f"band_conf={totals['band_conf']} ({totals['band_conf'] / n_in:.3f}), "
            f"rescue_kpt={totals['rescue_kpt']} ({totals['rescue_kpt'] / n_in:.3f}), "
            f"final={totals['final']} ({totals['final'] / n_in:.3f})"
        )
        if self._pp_last_thresholds:
            msg += (
                ", thresholds="
                f"base:{self._pp_last_thresholds.get('base_score_threshold', 0.0):.4f}, "
                f"local_min:{self._pp_last_thresholds.get('local_threshold_min', 0.0):.4f}, "
                f"local_max:{self._pp_last_thresholds.get('local_threshold_max', 0.0):.4f}, "
                f"band_ratio:[{self._pp_last_thresholds.get('band_low_ratio', 0.0):.3f},"
                f"{self._pp_last_thresholds.get('band_high_ratio', 0.0):.3f}], "
                f"num_bands:{int(self._pp_last_thresholds.get('num_bands', 0.0))}, "
                f"fallback:{self._pp_last_thresholds.get('fallback_threshold_ratio', 0.0):.3f}, "
                f"rescue_kpt:{self._pp_last_thresholds.get('rescue_keypoint_ratio', 0.0):.3f}"
            )
        LOGGER.info(msg)

        if self._pp_last_thresholds and self._pp_stage_eval_totals["input"] > 0:
            eval_totals = self._pp_stage_eval_totals
            eval_in = max(eval_totals["input"], 1)
            base_th = self._pp_last_thresholds.get("base_score_threshold", 0.0)
            LOGGER.info(
                "BEV calibrated-band post-process retention @conf>=base: "
                f"input={eval_totals['input']}, "
                f"band_conf={eval_totals['band_conf']} ({eval_totals['band_conf'] / eval_in:.3f}), "
                f"rescue_kpt={eval_totals['rescue_kpt']} ({eval_totals['rescue_kpt'] / eval_in:.3f}), "
                f"final={eval_totals['final']} ({eval_totals['final'] / eval_in:.3f}), "
                f"base={base_th:.4f}"
            )

    def _run_locsim(
        self,
        stats: dict,
        coco_gt: COCO,
        coco_dt,
        sigmas: list,
        iou_type: str,
        EvalClass,
        prefix: str,
    ) -> dict:
        """Run a single LocSim evaluation pass and merge results into stats.

        Mirrors mmpose coco_metric._do_python_keypoint_eval() lines 565-609.
        """
        # Both paths use the same name when phase="val":
        #   stats_file      = {prefix}_val_stats.json
        #   val_stats_file  = {prefix}_val_stats.json
        # We always write to val_stats_file so the test phase can load it.
        # For test/challenge we additionally write a phase-specific copy.
        val_stats_file = self.save_dir / f"{prefix}_val_stats.json"
        stats_file = self.save_dir / f"{prefix}_{self.phase}_stats.json"
        if self.bev_postprocessor is not None:
            stats_file = self.save_dir / f"{prefix}_{self.phase}_postprocess_stats.json"
        elif self._final_prediction_threshold is not None:
            stats_file = self.save_dir / f"{prefix}_{self.phase}_filtered_stats.json"

        coco_eval = EvalClass(coco_gt, coco_dt, "bbox", sigmas, use_area=True)

        # For test/challenge: load score_threshold from val run (mirrors mmpose test.py line 177)
        if self.bev_postprocessor is not None or self._final_prediction_threshold is not None:
            # In self-consistent mode, predictions.json already contains the
            # final filtered detections. Keep evaluator metrics aligned with
            # that set by using a threshold that all retained detections satisfy.
            eff_th = self._resolve_evaluator_threshold_for_filtered_predictions()
            coco_eval.params.score_threshold = eff_th
            LOGGER.info(
                f"Using runtime-filtered predictions for {prefix}; "
                f"setting evaluator score_threshold={eff_th:.4f} to match the retained prediction set."
            )
        elif self.phase in ("test", "challenge"):
            if not val_stats_file.exists():
                LOGGER.warning(
                    f"{val_stats_file} not found. "
                    "Run --split val first to obtain the optimal score threshold, "
                    "then re-run with --split test/challenge."
                )
            else:
                with open(val_stats_file) as f:
                    th = json.load(f)["stats"]["score_threshold"]
                LOGGER.info(f"Loaded score_threshold={th:.4f} from {val_stats_file}")
                coco_eval.params.score_threshold = th

        # position_from_keypoint_index=1 → pelvis_ground is the BEV position
        # (matches mmpose coco_metric.py line 581)
        coco_eval.params.position_from_keypoint_index = 1
        coco_eval.params.useSegm = None

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # LocSim stat names (from mmpose coco_metric.py lines 594-596)
        stats_names = [
            "AP", "AP .5", "AP .75", "AP (S)", "AP (M)", "AP (L)",
            "AR", "AR .5", "AR .75", "AR (S)", "AR (M)", "AR (L)",
            "precision", "recall", "f1", "score_threshold", "frame_accuracy",
        ]
        assert len(stats_names) == len(coco_eval.stats), (
            f"LocSim stats length mismatch: expected {len(stats_names)}, "
            f"got {len(coco_eval.stats)}"
        )

        result = dict(zip(stats_names, coco_eval.stats))

        if self.bev_postprocessor is not None or self._final_prediction_threshold is not None:
            # Preserve the original val-derived threshold file. In self-consistent
            # mode, phase-specific filtered stats should not overwrite the base
            # locsim_val_stats.json needed for future runs.
            with open(stats_file, "w") as f:
                json.dump({"stats": result}, f, indent=4)
            LOGGER.info(f"Saved filtered/postprocess stats → {stats_file}")
        else:
            # Always write val_stats_file so the test/challenge phase can load score_threshold.
            # When phase="val", stats_file IS val_stats_file — write once, no copy needed.
            with open(val_stats_file, "w") as f:
                json.dump({"stats": result}, f, indent=4)
            LOGGER.info(f"Saved val stats → {val_stats_file}")

            # For test/challenge, also write a separate phase-stamped copy for auditing.
            if self.phase != "val" and stats_file != val_stats_file:
                shutil.copyfile(val_stats_file, stats_file)

        LOGGER.info(f"\n{prefix.upper()} LocSim results ({self.phase}):")
        for k, v in result.items():
            LOGGER.info(f"  {k:>20s}: {v:.4f}" if isinstance(v, float) else f"  {k:>20s}: {v}")

        # Merge into main stats dict under namespaced keys
        for k, v in result.items():
            stats[f"{prefix}/{k}"] = v

        return stats
