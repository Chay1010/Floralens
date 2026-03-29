"""
FloraLens Model — ONNX Runtime inference engine for plant classification.

Loads the exported ONNX model at startup and provides a high-level predict() API.
Falls back to PyTorch if no ONNX model is found.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.config import FLOWER_NAMES, NUM_CLASSES, ONNX_MODEL_PATH

logger = logging.getLogger(__name__)


class FloraLensPredictor:
    """ONNX Runtime predictor for flower classification.

    Loads the model once on construction and reuses the session for all
    subsequent predictions to avoid per-request overhead (~150ms).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or ONNX_MODEL_PATH
        self.session = None
        self.input_name: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model into an InferenceSession."""
        import onnxruntime as ort

        if not self.model_path.exists():
            logger.warning(
                f"ONNX model not found at {self.model_path}. "
                "Run `python -m training.export_onnx` first. "
                "API will return 503 until model is available."
            )
            return

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        active_provider = self.session.get_providers()[0]
        logger.info(
            f"✅ ONNX model loaded from {self.model_path} "
            f"(provider: {active_provider})"
        )

    @property
    def is_ready(self) -> bool:
        return self.session is not None

    def predict(
        self, preprocessed_input: np.ndarray, top_k: int = 5
    ) -> List[dict]:
        """Run inference and return top-K predictions.

        Args:
            preprocessed_input: np.ndarray of shape (1, 3, 260, 260).
            top_k: Number of top predictions to return.

        Returns:
            List of dicts with keys: class_id, class_name, confidence.
        """
        if not self.is_ready:
            raise RuntimeError("Model not loaded. Export the ONNX model first.")

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: preprocessed_input})
        logits = outputs[0][0]  # shape: (NUM_CLASSES,)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()

        # Top-K
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "class_id": int(idx),
                "class_name": FLOWER_NAMES[idx] if idx < len(FLOWER_NAMES) else f"class_{idx}",
                "confidence": round(float(probabilities[idx]), 4),
            })

        return results

    def predict_with_latency(
        self, preprocessed_input: np.ndarray, top_k: int = 5
    ) -> Tuple[List[dict], float]:
        """Predict and also measure inference latency in milliseconds."""
        import time

        start = time.perf_counter()
        results = self.predict(preprocessed_input, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms


# Singleton — initialized at import time if model exists
predictor: Optional[FloraLensPredictor] = None


def get_predictor() -> FloraLensPredictor:
    """Get or create the global predictor singleton."""
    global predictor
    if predictor is None:
        predictor = FloraLensPredictor()
    return predictor
