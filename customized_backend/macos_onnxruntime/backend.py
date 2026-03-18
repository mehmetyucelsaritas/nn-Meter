import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults

logger = logging.getLogger("nn-Meter")


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.strip().lower() in {"1", "true", "yes", "y", "on"}:
            return True
        if v.strip().lower() in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _numpy_dtype_from_ort_type(ort_type: str) -> np.dtype:
    # Examples: "tensor(float)", "tensor(float16)", "tensor(double)", "tensor(int64)"
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type:
        return np.float32
    if "double" in ort_type:
        return np.float64
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "int8" in ort_type:
        return np.int8
    if "uint8" in ort_type:
        return np.uint8
    # Default for most exported kernel models
    return np.float32


def _make_inputs(
    ort_inputs: List[Any],
    input_shape: Optional[List[List[int]]],
    batch_size: int,
) -> Dict[str, np.ndarray]:
    feeds: Dict[str, np.ndarray] = {}
    for idx, ort_in in enumerate(ort_inputs):
        name = ort_in.name
        dtype = _numpy_dtype_from_ort_type(getattr(ort_in, "type", "tensor(float)"))

        # Prefer explicit shapes passed by nn-Meter (they exclude batch dim for torch impl)
        if input_shape and idx < len(input_shape):
            shape = [batch_size] + list(input_shape[idx])
        else:
            # Fallback to ORT model input shapes (may contain None / symbolic dims)
            raw = list(getattr(ort_in, "shape", [])) or [batch_size, 1]
            shape = [batch_size if (d is None or isinstance(d, str)) else int(d) for d in raw]

        feeds[name] = np.random.randn(*shape).astype(dtype, copy=False)
    return feeds


def _trim_times(times_ms: List[float], trim_ratio: float) -> List[float]:
    if not times_ms:
        return times_ms
    trim_ratio = max(0.0, min(0.49, float(trim_ratio)))
    if trim_ratio == 0.0:
        return times_ms
    xs = np.sort(np.asarray(times_ms, dtype=np.float64))
    k = int(len(xs) * trim_ratio)
    if len(xs) - 2 * k <= 0:
        return times_ms
    return xs[k : len(xs) - k].tolist()


@dataclass
class _ORTConfig:
    use_coreml_ep: bool
    intra_op_num_threads: int
    inter_op_num_threads: int
    warmup_runs: int
    num_runs: int
    repeats: int
    agg: str
    trim_ratio: float
    batch_size: int
    verbose: bool


class ORTLatencyParser(BaseParser):
    def __init__(self):
        self._latency = Latency()

    def parse(self, content: str):
        data = json.loads(content)
        self._latency = Latency(float(data["avg_ms"]), float(data["std_ms"]))
        return self

    @property
    def latency(self) -> Latency:
        return self._latency

    @property
    def results(self) -> ProfiledResults:
        return ProfiledResults({"latency": self.latency})


class ORTProfiler(BaseProfiler):
    def __init__(
        self,
        use_coreml_ep: bool = True,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
        warmup_runs: int = 10,
        num_runs: int = 50,
        repeats: int = 5,
        agg: str = "trimmed_mean",
        trim_ratio: float = 0.1,
        batch_size: int = 1,
        verbose: bool = False,
    ):
        self.cfg = _ORTConfig(
            use_coreml_ep=use_coreml_ep,
            intra_op_num_threads=intra_op_num_threads,
            inter_op_num_threads=inter_op_num_threads,
            warmup_runs=warmup_runs,
            num_runs=num_runs,
            repeats=repeats,
            agg=agg,
            trim_ratio=trim_ratio,
            batch_size=batch_size,
            verbose=verbose,
        )

    def _create_session(self, model_path: str):
        try:
            import onnxruntime as ort
        except Exception as e:
            raise ImportError(
                "onnxruntime is required for the macos_onnxruntime backend. "
                "Install it with `pip install onnxruntime` (and optionally a build with CoreML EP)."
            ) from e

        so = ort.SessionOptions()
        if self.cfg.intra_op_num_threads and self.cfg.intra_op_num_threads > 0:
            so.intra_op_num_threads = self.cfg.intra_op_num_threads
        if self.cfg.inter_op_num_threads and self.cfg.inter_op_num_threads > 0:
            so.inter_op_num_threads = self.cfg.inter_op_num_threads

        providers = []
        available = ort.get_available_providers()
        if self.cfg.use_coreml_ep and "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        if self.cfg.verbose:
            logger.info(f"[macos_onnxruntime] providers={sess.get_providers()} available={available}")
        return sess

    def profile(self, model_path: str, input_shape: Optional[List[List[int]]] = None, **kwargs) -> str:
        sess = self._create_session(model_path)
        ort_inputs = sess.get_inputs()
        feeds = _make_inputs(ort_inputs, input_shape=input_shape, batch_size=self.cfg.batch_size)

        # Warmups
        for _ in range(max(0, self.cfg.warmup_runs)):
            _ = sess.run(None, feeds)

        # Timed runs (repeat rounds to reduce noise)
        all_times: List[float] = []
        round_avgs: List[float] = []
        repeats = max(1, int(self.cfg.repeats))
        runs = max(1, int(self.cfg.num_runs))
        for _ in range(repeats):
            times_ms: List[float] = []
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = sess.run(None, feeds)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)
            trimmed = _trim_times(times_ms, self.cfg.trim_ratio) if self.cfg.agg == "trimmed_mean" else times_ms
            round_avgs.append(float(np.mean(trimmed)) if trimmed else float(np.mean(times_ms)))
            all_times.extend(times_ms)

        agg = (self.cfg.agg or "").strip().lower()
        if agg == "median":
            avg_ms = float(np.median(round_avgs))
        elif agg == "mean":
            avg_ms = float(np.mean(round_avgs))
        else:  # "trimmed_mean" (default)
            avg_ms = float(np.mean(round_avgs))

        # Std reported over per-run times (after trimming only when enabled)
        if agg == "trimmed_mean":
            used = _trim_times(all_times, self.cfg.trim_ratio)
        else:
            used = all_times
        std_ms = float(np.std(np.asarray(used, dtype=np.float64), ddof=0)) if used else 0.0
        return json.dumps(
            {
                "avg_ms": avg_ms,
                "std_ms": std_ms,
                "num_runs": runs,
                "repeats": repeats,
                "agg": agg,
                "trim_ratio": float(self.cfg.trim_ratio),
            }
        )


class MacOSONNXRuntimeBackend(BaseBackend):
    """
    Custom backend for profiling ONNX models locally on macOS (Apple Silicon friendly).

    Best used with nn-Meter builder config:
      - predictorbuild_config.yaml: IMPLEMENT: torch
    """

    parser_class = ORTLatencyParser
    profiler_class = ORTProfiler

    def update_configs(self):
        super().update_configs()
        cfg = self.configs or {}
        self.profiler_kwargs.update(
            {
                "use_coreml_ep": _safe_bool(cfg.get("USE_COREML_EP", True), True),
                "intra_op_num_threads": _safe_int(cfg.get("INTRA_OP_NUM_THREADS", 0), 0),
                "inter_op_num_threads": _safe_int(cfg.get("INTER_OP_NUM_THREADS", 0), 0),
                "warmup_runs": _safe_int(cfg.get("WARMUP_RUNS", 25), 25),
                "num_runs": _safe_int(cfg.get("NUM_RUNS", 100), 100),
                "repeats": _safe_int(cfg.get("REPEATS", 5), 5),
                "agg": str(cfg.get("AGG", "trimmed_mean")),
                "trim_ratio": _safe_float(cfg.get("TRIM_RATIO", 0.1), 0.1),
                "batch_size": _safe_int(cfg.get("BATCH_SIZE", 1), 1),
                "verbose": _safe_bool(cfg.get("VERBOSE", False), False),
            }
        )

    def convert_model(self, model_path, save_path, input_shape=None):
        # For torch implementation mode, nn-Meter already exports ONNX models.
        # If a path endswith ".onnx", just pass it through.
        if isinstance(model_path, str) and model_path.endswith(".onnx"):
            return model_path
        raise NotImplementedError(
            "macos_onnxruntime backend expects an ONNX model. "
            "In your workspace set `IMPLEMENT: torch` so nn-Meter exports kernels to ONNX."
        )

    def test_connection(self):
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            logger.info(f"hello backend ! (onnxruntime providers: {providers})")
        except Exception as e:
            raise ImportError(
                "onnxruntime is required for the macos_onnxruntime backend. Install it with `pip install onnxruntime`."
            ) from e

