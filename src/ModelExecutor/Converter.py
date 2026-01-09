"""
Utility to convert a Keras model to an INT8-quantized TensorFlow Lite model.

Example:
    python convert_to_tflite.py --model path/to/model.keras \
        --representative-data path/to/rep_data.npy \
        --output harfm_int8.tflite --num-samples 200

Notes:
    - Representative data should have shape [N, time, channels] (or [N, time]
      which will be expanded to a single channel). Calibration works best when
      these samples reflect real inputs, not random noise.
    - If representative data is omitted, a small random set is used. Provide
      --sequence-length when doing so to fix the time dimension.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable, Optional, Union

import numpy as np
import tensorflow as tf


def _create_tflite_converter(
    model: tf.keras.Model,
    *,
    use_concrete_function: bool = False,
) -> tf.lite.TFLiteConverter:
    """Return a TFLite converter, optionally forcing the concrete-function path."""

    if not use_concrete_function:
        return tf.lite.TFLiteConverter.from_keras_model(model)

    input_specs = []
    for tensor in model.inputs:
        if tensor.shape is None:
            raise ValueError("All model inputs must have a defined shape for TFLite export.")
        input_specs.append(tf.TensorSpec(shape=tensor.shape, dtype=tensor.dtype))

    concrete_fn = tf.function(model, jit_compile=False).get_concrete_function(*input_specs)
    return tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], model)


def _get_model_input_dims(model: tf.keras.Model) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (time_dim, channels, extra_dim) for a single-input model."""

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError("Only single-input models are supported for TFLite conversion.")
        input_shape = input_shape[0]
    if isinstance(input_shape, tf.TensorShape):
        input_shape = tuple(input_shape.as_list())

    if len(input_shape) == 3:
        _, time_dim, channels = input_shape
        extra_dim = None
    elif len(input_shape) == 4:
        _, time_dim, channels, extra_dim = input_shape
    else:
        raise ValueError(
            f"Unsupported input rank {len(input_shape)}; only 3D or 4D inputs are supported."
        )
    return time_dim, channels, extra_dim


def _ensure_sample_shape(
    samples: np.ndarray,
    *,
    channels: int,
    extra_dim: Optional[int],
) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim == 2:
        samples = samples[:, :, np.newaxis]

    if extra_dim is not None:
        if samples.ndim == 3:
            if extra_dim not in (1, None):
                raise ValueError(
                    "Representative samples are missing the final dimension required by the model."
                )
            samples = samples[..., np.newaxis]
        if samples.ndim != 4:
            raise ValueError(
                f"Representative samples must have 4 dims (N, time, channels, extra), got {samples.shape}."
            )
        if extra_dim is not None and samples.shape[3] != extra_dim:
            raise ValueError(
                f"Expected representative samples with last dimension {extra_dim}, got {samples.shape[3]}."
            )
    else:
        if samples.ndim != 3:
            raise ValueError(
                f"Representative samples must have 3 dims (N, time, channels), got {samples.shape}."
            )

    if samples.shape[2] != channels:
        raise ValueError(f"Expected {channels} channels, but samples have {samples.shape[2]}.")
    return samples


def _load_representative_array(
    path: pathlib.Path,
    *,
    key: Optional[str],
    seq_len: Optional[int],
    channels: Optional[int],
    extra_dim: Optional[int],
    num_samples: int,
) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if key is None:
            raise ValueError("For .npz files, provide --npz-key to select the array.")
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
        arr = data[key]
    else:
        arr = np.load(path)

    if channels is None:
        raise ValueError("Model input channels are undefined; cannot validate representative data.")

    arr = _ensure_sample_shape(arr, channels=channels, extra_dim=extra_dim)

    if seq_len is not None:
        if arr.shape[1] < seq_len:
            pad_shape = (arr.shape[0], seq_len - arr.shape[1]) + tuple(arr.shape[2:])
            pad = np.zeros(pad_shape, dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > seq_len:
            arr = arr[:, :seq_len, ...]

    arr = arr.astype(np.float32)
    return arr[:num_samples]


def _make_representative_dataset(samples: np.ndarray) -> Iterable[list[np.ndarray]]:
    for sample in samples:
        yield [sample[np.newaxis, ...]]


def _random_representative(
    *,
    seq_len: int,
    channels: int,
    num_samples: int,
    scale: float = 1.0,
    extra_dim: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    base_shape = (num_samples, seq_len, channels)
    if extra_dim is not None:
        base_shape = base_shape + (extra_dim,)
    return (rng.standard_normal(base_shape) * scale).astype(np.float32)


def convert_model_to_tflite_bytes(
    model: tf.keras.Model,
    *,
    representative_samples: Optional[np.ndarray],
    num_samples: int = 128,
    sequence_length: Optional[int] = None,
    quantization_mode: str = "int8",
) -> bytes:
    """Convert an in-memory Keras model to TFLite bytes (INT8 or float16)."""
    time_dim, channels, extra_dim = _get_model_input_dims(model)
    if channels is None:
        raise ValueError("Model input channels are undefined; please set them when building the model.")

    mode = quantization_mode.lower()
    if mode not in {"int8", "float16"}:
        raise ValueError("quantization_mode must be 'int8' or 'float16'.")

    samples: Optional[np.ndarray] = None
    if mode == "int8":
        print("[convert] Using INT8 quantization.")
        if representative_samples is not None:
            print("[convert] Preparing representative samples array.")
            samples = _ensure_sample_shape(
                representative_samples,
                channels=channels,
                extra_dim=extra_dim,
            )
            seq_len = sequence_length or samples.shape[1]
            if samples.shape[1] < seq_len:
                pad_shape = (samples.shape[0], seq_len - samples.shape[1]) + tuple(samples.shape[2:])
                pad = np.zeros(pad_shape, dtype=samples.dtype)
                samples = np.concatenate([samples, pad], axis=1)
            elif samples.shape[1] > seq_len:
                samples = samples[:, :seq_len, ...]
            samples = samples.astype(np.float32)[:num_samples]
        else:
            print("[convert] No representative samples provided; generating random calibration data.")
            if time_dim is None and sequence_length is None:
                raise ValueError("Provide sequence_length when the model has a dynamic time dimension.")
            seq_len = sequence_length or time_dim
            samples = _random_representative(
                seq_len=seq_len,
                channels=channels,
                num_samples=num_samples,
                extra_dim=extra_dim,
            )
    else:
        print("[convert] Using float16 quantization.")

    def _configure_converter(conv: tf.lite.TFLiteConverter) -> tf.lite.TFLiteConverter:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if mode == "int8":
            assert samples is not None
            conv.representative_dataset = lambda: _make_representative_dataset(samples)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        else:
            conv.target_spec.supported_types = [tf.float16]
            # Keep inference input/output as float32 for widest compatibility; weights become float16.
        return conv

    def _run_conversion(conv: tf.lite.TFLiteConverter) -> bytes:
        print("[convert] Starting converter.convert() ...")
        result_inner = conv.convert()
        print("[convert] Finished converter.convert().")
        return result_inner

    converter = _configure_converter(_create_tflite_converter(model))

    try:
        return _run_conversion(converter)
    except AttributeError as exc:
        if "_get_save_spec" not in str(exc):
            raise
        print(
            "[convert] tf.keras model is missing _get_save_spec; retrying with concrete"
            " function conversion."
        )
        converter = _configure_converter(
            _create_tflite_converter(model, use_concrete_function=True)
        )
        return _run_conversion(converter)


def _sanitize_symbol_name(name: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not sanitized:
        sanitized = "model"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _write_platformio_source(data: bytes, path: pathlib.Path, *, symbol_hint: str = "g_model_tflm") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = _sanitize_symbol_name(symbol_hint or path.stem)
    array_name = base if base.endswith("_data") else f"{base}_data"
    length_name = f"{array_name}_len"

    with path.open("w", encoding="utf-8") as f:
        f.write("// Auto-generated by convert_to_tflite. Do not edit.\n")
        f.write("#include <cstddef>\n\n")
        f.write(f"alignas(16) extern const unsigned char {array_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 12])
            f.write(f"  {chunk},\n")
        f.write("};\n")
        f.write(f"extern const int {length_name} = {len(data)};\n")


def convert_to_tflite(
    model_or_path: Union[str, os.PathLike, pathlib.Path, tf.keras.Model],
    output_path: Union[str, os.PathLike, pathlib.Path],
    *,
    representative_data: Optional[Union[pathlib.Path, np.ndarray, tf.Tensor]],
    npz_key: Optional[str],
    num_samples: int,
    sequence_length: Optional[int],
    quantization_mode: str = "int8",
    output_path_pio: Optional[Union[str, os.PathLike, pathlib.Path]] = None,
):
    def _to_path(p: Union[str, os.PathLike, pathlib.Path]) -> pathlib.Path:
        return p if isinstance(p, pathlib.Path) else pathlib.Path(p)

    def _load_model(m: Union[str, os.PathLike, pathlib.Path, tf.keras.Model]) -> tf.keras.Model:
        if isinstance(m, tf.keras.Model):
            return m
        return tf.keras.models.load_model(_to_path(m))

    def _prepare_samples(
        rep: Optional[Union[pathlib.Path, np.ndarray]],
        *,
        channels: int,
        time_dim: Optional[int],
        extra_dim: Optional[int],
    ) -> np.ndarray:
        if rep is None:
            print("[convert_to_tflite] No representative data provided; will use random samples if needed.")
            if time_dim is None and sequence_length is None:
                raise ValueError("Provide sequence_length when the model has a dynamic time dimension.")
            seq_len = sequence_length or time_dim
            print(
                "No representative data provided; using random samples for calibration. "
                "Supply real data for better accuracy.",
                file=sys.stderr,
            )
            return _random_representative(
                seq_len=seq_len,
                channels=channels,
                num_samples=num_samples,
                extra_dim=extra_dim,
            )

        if isinstance(rep, (str, pathlib.Path)):
            print(f"[convert_to_tflite] Loading representative data from {rep}.")
            rep_path = pathlib.Path(rep)
            return _load_representative_array(
                rep_path,
                key=npz_key,
                seq_len=sequence_length,
                channels=channels,
                extra_dim=extra_dim,
                num_samples=num_samples,
            )

        # Assume array/tensor-like input
        print("[convert_to_tflite] Using in-memory representative data.")
        samples = _ensure_sample_shape(rep, channels=channels, extra_dim=extra_dim)
        seq_len = sequence_length or samples.shape[1]
        if samples.shape[1] < seq_len:
            pad_shape = (samples.shape[0], seq_len - samples.shape[1]) + tuple(samples.shape[2:])
            pad = np.zeros(pad_shape, dtype=samples.dtype)
            samples = np.concatenate([samples, pad], axis=1)
        elif samples.shape[1] > seq_len:
            samples = samples[:, :seq_len, ...]
        return samples.astype(np.float32)[:num_samples]

    model = _load_model(model_or_path)
    print("[convert_to_tflite] Model loaded.")

    time_dim, channels, extra_dim = _get_model_input_dims(model)
    if channels is None:
        raise ValueError("Model input channels are undefined; please set them when building the model.")

    mode = quantization_mode.lower()
    if mode == "int8":
        rep_samples = _prepare_samples(
            representative_data,
            channels=channels,
            time_dim=time_dim,
            extra_dim=extra_dim,
        )
    else:
        rep_samples = None

    tflite_model = convert_model_to_tflite_bytes(
        model,
        representative_samples=rep_samples,
        num_samples=num_samples,
        sequence_length=sequence_length,
        quantization_mode=quantization_mode,
    )
    _to_path(output_path).write_bytes(tflite_model)
    print(f"Saved INT8 TFLite model to {output_path}")
    if output_path_pio is not None:
        pio_path = _to_path(output_path_pio)
        symbol_hint = pio_path.stem
        _write_platformio_source(tflite_model, pio_path, symbol_hint=symbol_hint)
        print(f"Wrote PlatformIO source to {pio_path}")
    return tflite_model


def main():
    parser = argparse.ArgumentParser(description="Convert a Keras model to INT8 TFLite.")
    parser.add_argument("--model", required=True, type=pathlib.Path, help="Path to Keras model (.keras, .h5, or SavedModel dir).")
    parser.add_argument("--output", type=pathlib.Path, help="Output .tflite path. Defaults to <model>.tflite next to the input model.")
    parser.add_argument("--representative-data", type=pathlib.Path, help="Path to .npy or .npz file with calibration data.")
    parser.add_argument("--npz-key", type=str, help="Key inside an .npz file to load as representative data.")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of samples to use for calibration.")
    parser.add_argument("--sequence-length", type=int, help="Fix the time dimension when using random calibration data.")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int8", "float16"],
        default="int8",
        help="Quantization mode. INT8 uses representative data; float16 ignores it.",
    )
    parser.add_argument(
        "--output-path-pio",
        type=pathlib.Path,
        help="Optional .cc path to emit the model as a constexpr byte array for PlatformIO builds.",
    )

    args = parser.parse_args()

    model_path = args.model
    output_path = args.output or model_path.with_suffix(".tflite")

    convert_to_tflite(
        model_or_path=model_path,
        output_path=output_path,
        representative_data=args.representative_data,
        npz_key=args.npz_key,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        quantization_mode=args.quantization,
        output_path_pio=args.output_path_pio,
    )


if __name__ == "__main__":
    main()
