"""Helpers for inspecting/exporting TFLite models for firmware usage."""

from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import tflite


BytesLike = Union[bytes, bytearray, memoryview]
_BYTES_TYPES = (bytes, bytearray, memoryview)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolver_header_path() -> Path:
    return _project_root() / "firmware" / "lib" / "tflm" / "tensorflow" / "lite" / "micro" / "micro_mutable_op_resolver.h"


def _register_operators_cpp_path() -> Path:
    return _project_root() / "firmware" / "src" / "register_operators.cpp"


def _register_operators_header_path() -> Path:
    return _project_root() / "firmware" / "src" / "register_operators.h"


def _read_file(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError as err:
        raise RuntimeError(f"Unable to locate resolver header at {path}.") from err


def _extract_block(text: str, start_index: int) -> Tuple[str, int]:
    depth = 0
    for idx in range(start_index, len(text)):
        char = text[idx]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start_index + 1 : idx], idx + 1
    raise ValueError("Failed to extract block; unmatched braces detected in resolver header.")


@functools.lru_cache(maxsize=1)
def _builtin_to_method_map() -> Dict[str, str]:
    header_text = _read_file(_resolver_header_path())
    results: Dict[str, str] = {}
    pattern = re.compile(r"TfLiteStatus\s+(Add\w+)\s*\([^)]*\)")
    for match in pattern.finditer(header_text):
        method_name = match.group(1)
        block_start = header_text.find('{', match.end())
        if block_start == -1:
            continue
        body, _ = _extract_block(header_text, block_start)
        builtin_match = re.search(r"AddBuiltin\s*\(\s*BuiltinOperator_(\w+)", body)
        if builtin_match:
            builtin_name = builtin_match.group(1)
            results.setdefault(builtin_name, method_name)
        # Skip custom registrations that don't call AddBuiltin.
    return results


def _load_model_bytes(tflite_model: Union[str, os.PathLike, BytesLike]) -> bytes:
    if isinstance(tflite_model, (str, os.PathLike)):
        return Path(tflite_model).read_bytes()
    if isinstance(tflite_model, _BYTES_TYPES):
        return bytes(tflite_model)
    raise TypeError("tflite_model must be a file path or raw bytes-like object.")


def register_ops_tflite(tflite_model: Union[str, os.PathLike, BytesLike]) -> Dict[int, str]:
    builtin_map = _builtin_to_method_map()
    model_bytes = _load_model_bytes(tflite_model)
    tf_model = tflite.Model.GetRootAsModel(model_bytes, 0)

    ops: List[Tuple[int, str, str]] = []
    seen_codes: set[int] = set()
    for idx in range(tf_model.OperatorCodesLength()):
        op_code = tf_model.OperatorCodes(idx)
        builtin_code = op_code.BuiltinCode()
        if builtin_code < 0:
            continue
        op_name = tflite.opcode2name(builtin_code)
        if op_name == "CUSTOM":
            custom_name = op_code.CustomCode()
            custom_name = custom_name.decode("utf-8") if custom_name else "<unknown>"
            raise NotImplementedError(
                f"Custom operator '{custom_name}' encountered; automatic resolver generation not supported."
            )
        method_name = builtin_map.get(op_name)
        if method_name is None:
            raise KeyError(
                f"Unsupported builtin operator '{op_name}' encountered; update the resolver mapping parser."
            )
        if builtin_code not in seen_codes:
            ops.append((int(builtin_code), op_name, method_name))
            seen_codes.add(builtin_code)

    ops.sort(key=lambda item: item[0])
    _write_register_operator_files(ops)
    return {code: method for code, _, method in ops}


def _write_register_operator_files(ops: Sequence[Tuple[int, str, str]]) -> None:
    cpp_path = _register_operators_cpp_path()
    header_path = _register_operators_header_path()
    cpp_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.parent.mkdir(parents=True, exist_ok=True)

    header_content = _render_register_operators_header(ops)
    cpp_content = _render_register_operators_cpp(ops)
    _write_text_if_changed(header_path, header_content)
    _write_text_if_changed(cpp_path, cpp_content)


def _write_text_if_changed(path: Path, content: str) -> None:
    try:
        current = path.read_text()
    except FileNotFoundError:
        current = None
    if current == content:
        return
    path.write_text(content)


def _write_bytes_if_changed(path: Path, content: bytes) -> None:
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    if current == content:
        return
    path.write_bytes(content)


def _render_register_operators_header(ops: Sequence[Tuple[int, str, str]]) -> str:
    resolver_min = max(1, len(ops))
    lines = ["// Auto-generated by register_ops_tflite(). Do not edit.", "#pragma once",
             "", "#include \"tensorflow/lite/core/c/common.h\"",
             "#include \"tensorflow/lite/micro/micro_mutable_op_resolver.h\"", "",
             "namespace tinyfm {", f"constexpr int kResolverOpCount = {resolver_min};",
             "using GeneratedOpResolver = tflite::MicroMutableOpResolver<kResolverOpCount>;",
             "", "TfLiteStatus RegisterGeneratedOps(GeneratedOpResolver& op_resolver);",
             "", "}  // namespace tinyfm", ""]
    return "\n".join(lines)


def _render_register_operators_cpp(ops: Sequence[Tuple[int, str, str]]) -> str:
    lines = ["// Auto-generated by register_ops_tflite(). Do not edit.",
             "#include \"register_operators.h\"",
             "#include \"tensorflow/lite/kernels/op_macros.h\"", "",
             "namespace tinyfm {", "",
             "TfLiteStatus RegisterGeneratedOps(GeneratedOpResolver& op_resolver) {"]
    if ops:
        for code, name, method in ops:
            lines.append(f"  // BuiltinOperator_{name} ({code})")
            lines.append(f"  TF_LITE_ENSURE_STATUS(op_resolver.{method}());")
    else:
        lines.append("  // No operators present in the model.")
    lines.append("  return kTfLiteOk;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace tinyfm")
    lines.append("")
    return "\n".join(lines)


def execute_model(
    tflite_model: Optional[Union[str, os.PathLike, BytesLike]] = None,
    *,
    run_upload: bool = True,
    run_monitor: bool = True,
    port: Optional[str] = None,
    env: Optional[str] = None,
) -> None:
    """Generate firmware assets for a TFLite model and run it on the MCU."""
    firmware_root = _project_root() / "firmware"
    firmware_src = firmware_root / "src"
    firmware_src.mkdir(parents=True, exist_ok=True)

    if tflite_model is None:
        model_source = (firmware_src / "model.tflite").resolve()
        if not model_source.exists():
            raise FileNotFoundError(
                "tflite_model not provided and no model.tflite found in firmware/src."
            )
        model_bytes = model_source.read_bytes()
    else:
        model_source = (
            Path(tflite_model).resolve()
            if isinstance(tflite_model, (str, os.PathLike))
            else None
        )
        model_bytes = _load_model_bytes(tflite_model)

    model_tflite_path = firmware_src / "model.tflite"
    if model_source is None or model_source != model_tflite_path.resolve():
        _write_bytes_if_changed(model_tflite_path, model_bytes)

    register_ops_tflite(model_bytes)

    from .Converter import _write_platformio_source

    _write_platformio_source(model_bytes, firmware_src / "model_data.cc", symbol_hint="g_model_tflm")
    _write_text_if_changed(
        firmware_src / "model_data.h",
        "#pragma once\n#include <cstddef>\n\n"
        "extern const unsigned char g_model_tflm_data[];\n"
        "extern const int g_model_tflm_data_len;\n",
    )

    pio_cmd = ["pio", "run", "-d", str(firmware_root)]
    if env:
        pio_cmd.extend(["-e", env])
    if run_upload:
        pio_cmd.extend(["-t", "upload"])
    if run_monitor:
        pio_cmd.extend(["-t", "monitor"])
    if port:
        pio_cmd.extend(["--upload-port", port])
        if run_monitor:
            pio_cmd.extend(["--monitor-port", port])

    subprocess.run(pio_cmd, check=True)
