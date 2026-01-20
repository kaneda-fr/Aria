"""Parakeet TDT inference.

ARIA supports two local model layouts:

1) A single waveform-input `.onnx` file under `models/parakeet-tdt-0.6b-v3/`.
    In this case we run the model directly via ONNX Runtime.

2) The repo's checked-in `models/parakeet-tdt-0.6b-v3-onnx/` directory which is
    intended to be used via the `onnx-asr` library.

The WebSocket server relies on `ParakeetOnnx.transcribe(pcm16_bytes)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort


class _Backend:
    ORT_WAVEFORM = "ort_waveform"
    ONNX_ASR = "onnx_asr"


@dataclass(frozen=True, slots=True)
class OnnxValueInfo:
    name: str
    type: str
    shape: tuple[Any, ...]


class ParakeetOnnx:
    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
    ) -> None:
        """Initialize Parakeet backend (CPU-only).

        Threading defaults are controlled by env vars:
        - ORT_INTRA_OP
        - ORT_INTER_OP
        """

        self._backend: str

        # Prefer an explicit `.onnx` file path when provided.
        resolved_model_path: Path | None = None
        if model_path is not None:
            resolved_model_path = self._resolve_model_path(model_path)

        # Otherwise, try the single-file waveform ONNX layout.
        if resolved_model_path is None:
            try:
                resolved_model_path = self._resolve_model_path(None)
            except Exception:
                resolved_model_path = None

        if resolved_model_path is not None:
            self._backend = _Backend.ORT_WAVEFORM
            sess_options = ort.SessionOptions()
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            intra = intra_op_threads if intra_op_threads is not None else _env_int("ORT_INTRA_OP")
            inter = inter_op_threads if inter_op_threads is not None else _env_int("ORT_INTER_OP")
            if intra is None:
                intra = os.cpu_count() or 1
            if inter is None:
                inter = 1
            sess_options.intra_op_num_threads = int(intra)
            sess_options.inter_op_num_threads = int(inter)

            self._session = ort.InferenceSession(
                str(resolved_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._inputs = tuple(_value_info(i) for i in self._session.get_inputs())
            self._outputs = tuple(_value_info(o) for o in self._session.get_outputs())
            self._onnx_asr = None
            return

        # Fallback: use the repo's `parakeet-tdt-0.6b-v3-onnx/` directory via onnx-asr.
        models_base = _models_base_dir()
        model_dir = models_base / "parakeet-tdt-0.6b-v3-onnx"
        if not model_dir.exists():
            raise FileNotFoundError(
                "No local Parakeet model found. Expected either a single .onnx under "
                f"{(models_base / 'parakeet-tdt-0.6b-v3')} or an onnx-asr model dir at {model_dir}."
            )

        import onnx_asr  # lazy import to avoid hard dependency when using the ORT single-file backend

        self._backend = _Backend.ONNX_ASR
        self._session = None
        self._inputs = tuple()
        self._outputs = tuple()
        self._onnx_asr = onnx_asr.load_model(
            "nemo-parakeet-tdt-0.6b-v3",
            str(model_dir),
            providers=["CPUExecutionProvider"],
        )

    @property
    def inputs(self) -> tuple[OnnxValueInfo, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[OnnxValueInfo, ...]:
        return self._outputs

    def describe_io(self) -> dict[str, list[dict[str, Any]]]:
        """Return a JSON-serializable description of model inputs/outputs."""

        return {
            "inputs": [
                {"name": i.name, "type": i.type, "shape": list(i.shape)} for i in self._inputs
            ],
            "outputs": [
                {"name": o.name, "type": o.type, "shape": list(o.shape)} for o in self._outputs
            ],
        }

    def infer(self, feeds: Mapping[str, Any]) -> dict[str, Any]:
        """Run a single ONNX inference.

        `feeds` must include all required model inputs.
        Returns a dict mapping output name -> output value.
        """

        if self._backend != _Backend.ORT_WAVEFORM or self._session is None:
            raise RuntimeError("infer() is only supported for the single-file ONNX backend")

        output_names = [o.name for o in self._outputs]
        results = self._session.run(output_names, dict(feeds))
        return dict(zip(output_names, results, strict=True))

    def transcribe(
        self,
        pcm16: bytes | np.ndarray,
        *,
        sample_rate: int = 16000,
        extra_inputs: Mapping[str, Any] | None = None,
    ) -> str:
        """Transcribe PCM16LE mono audio.

        This currently supports only models that accept a single waveform input.
        If your model requires additional inputs (features, lengths, etc.), pass
        them explicitly via `extra_inputs`.
        """

        if sample_rate != 16000:
            raise ValueError(f"ARIA expects 16000 Hz audio, got {sample_rate}")

        if self._backend == _Backend.ONNX_ASR:
            if isinstance(pcm16, (bytes, bytearray, memoryview)):
                waveform_i16 = np.frombuffer(pcm16, dtype=np.int16)
            else:
                waveform_i16 = np.asarray(pcm16)
            if waveform_i16.dtype != np.int16:
                raise TypeError(f"pcm16 must be int16, got {waveform_i16.dtype}")
            waveform_f32 = waveform_i16.astype(np.float32) / 32768.0

            result = self._onnx_asr.recognize(waveform_f32, sample_rate=16000)
            if isinstance(result, str):
                return result
            # onnx-asr can return structured results for some adapters; keep it safe.
            return str(result)

        if isinstance(pcm16, (bytes, bytearray, memoryview)):
            waveform_i16 = np.frombuffer(pcm16, dtype=np.int16)
        else:
            waveform_i16 = np.asarray(pcm16)

        if waveform_i16.dtype != np.int16:
            raise TypeError(f"pcm16 must be int16, got {waveform_i16.dtype}")

        waveform_f32 = waveform_i16.astype(np.float32) / 32768.0
        feeds: dict[str, Any] = {}

        if len(self._inputs) == 0:
            raise RuntimeError("Model has no inputs")

        if len(self._inputs) == 1:
            inp = self._inputs[0]
            feeds[inp.name] = _coerce_waveform_to_input(waveform_f32, inp)
        else:
            # Avoid guessing: require caller to provide required non-waveform inputs.
            if not extra_inputs:
                raise RuntimeError(
                    "Model requires multiple inputs; pass `extra_inputs` explicitly. "
                    f"I/O schema: {self.describe_io()}"
                )

            # Try to fill a waveform input if there is an obvious candidate.
            waveform_input = _select_waveform_input(self._inputs)
            if waveform_input is None:
                raise RuntimeError(
                    "Could not identify a waveform input among model inputs; "
                    f"I/O schema: {self.describe_io()}"
                )

            feeds[waveform_input.name] = _coerce_waveform_to_input(waveform_f32, waveform_input)
            feeds.update(dict(extra_inputs))

        outputs = self.infer(feeds)
        text = _extract_text_output(outputs)
        if text is None:
            raise RuntimeError(
                "Model outputs are not directly decodable to text yet (decoder not implemented). "
                f"Outputs: {[(k, _summarize_value(v)) for k, v in outputs.items()]}"
            )
        return text

    @staticmethod
    def _resolve_model_path(model_path: str | os.PathLike[str] | None) -> Path:
        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(str(path))
            return path

        models_base = _models_base_dir()
        model_dir = models_base / "parakeet-tdt-0.6b-v3"
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}. "
                "Place the Parakeet ONNX model there or pass model_path explicitly."
            )

        candidates = sorted(model_dir.glob("*.onnx"))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No .onnx files found under {model_dir}. "
                "Download/copy the model there or pass model_path explicitly."
            )
        raise FileExistsError(
            f"Multiple .onnx files found under {model_dir}: {candidates}. "
            "Pass model_path explicitly."
        )


def _models_base_dir() -> Path:
    """Return the base directory containing `models/`.

    For installed deployments, the repo checkout is not available inside site-packages.
    Prefer explicit configuration:

    - `ARIA_MODELS_DIR`: absolute/relative path to the directory containing model folders.

    Fallback:
    - `./models` relative to the current working directory.
    """

    env = os.environ.get("ARIA_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "models").resolve()


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Invalid int in env var {name}={value!r}") from e


def _value_info(v: Any) -> OnnxValueInfo:
    shape = getattr(v, "shape", None)
    if shape is None:
        shape_tuple: tuple[Any, ...] = tuple()
    else:
        shape_tuple = tuple(shape)

    return OnnxValueInfo(name=v.name, type=v.type, shape=shape_tuple)


def _select_waveform_input(inputs: tuple[OnnxValueInfo, ...]) -> OnnxValueInfo | None:
    # Conservative: pick a single float tensor input if there is exactly one.
    float_candidates = [i for i in inputs if i.type in {"tensor(float)", "tensor(float16)"}]
    if len(float_candidates) == 1:
        return float_candidates[0]
    return None


def _coerce_waveform_to_input(waveform_f32: np.ndarray, inp: OnnxValueInfo) -> np.ndarray:
    if inp.type not in {"tensor(float)", "tensor(float16)"}:
        raise RuntimeError(
            f"Waveform input must be float tensor, got {inp.name}:{inp.type} (shape={inp.shape})"
        )

    arr = waveform_f32
    rank = len(inp.shape)
    if rank == 1:
        return arr.astype(np.float32, copy=False)
    if rank == 2:
        # Common pattern: [B, T]
        return arr[np.newaxis, :].astype(np.float32, copy=False)

    raise RuntimeError(
        f"Unsupported waveform input rank for {inp.name}: shape={inp.shape}. "
        "Provide your own feature pipeline and call infer() directly."
    )


def _extract_text_output(outputs: Mapping[str, Any]) -> str | None:
    # If the model emits a string tensor output, return the first element.
    for value in outputs.values():
        if isinstance(value, (str, bytes)):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

        if isinstance(value, np.ndarray):
            if value.dtype.kind in {"U", "S", "O"}:
                if value.size == 0:
                    continue
                first = value.reshape(-1)[0]
                if isinstance(first, bytes):
                    return first.decode("utf-8", errors="replace")
                if isinstance(first, str):
                    return first
    return None


def _summarize_value(v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray(dtype={v.dtype}, shape={v.shape})"
    return f"{type(v).__name__}"
