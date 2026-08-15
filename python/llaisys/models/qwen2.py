import json
from ctypes import c_int, c_int64, c_size_t, c_void_p
from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open

from ..libllaisys import DataType, DeviceType, LIB_LLAISYS
from ..libllaisys.qwen2 import LlaisysQwen2Meta, Qwen2WeightLoadStatus


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen2 config not found: {config_path}")
        weight_files = sorted(model_path.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No safetensors files found in: {model_path}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        dtype = self._config_dtype(config["torch_dtype"])
        hidden_size = int(config["hidden_size"])
        num_heads = int(config["num_attention_heads"])
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        end_token = config["eos_token_id"]
        if isinstance(end_token, list):
            if len(end_token) != 1:
                raise ValueError("LLAISYS currently supports exactly one Qwen2 EOS token")
            end_token = end_token[0]

        self._meta = LlaisysQwen2Meta(
            dtype=int(dtype),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=num_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=hidden_size // num_heads,
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config["rope_theta"]),
            end_token=int(end_token),
        )
        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            self._meta, int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        try:
            for file in weight_files:
                with safe_open(file, framework="pt", device="cpu") as weights:
                    for name in weights.keys():
                        tensor = weights.get_tensor(name).contiguous()
                        self._load_weight(name, tensor)

            expected = int(
                LIB_LLAISYS.llaisysQwen2ModelExpectedWeightCount(self._model)
            )
            loaded = int(
                LIB_LLAISYS.llaisysQwen2ModelLoadedWeightCount(self._model)
            )
            if loaded != expected:
                raise RuntimeError(
                    f"Incomplete Qwen2 checkpoint: loaded {loaded} of {expected} weights"
                )
        except Exception:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None
            raise

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    @staticmethod
    def _config_dtype(name: str) -> DataType:
        dtypes = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }
        try:
            return dtypes[name]
        except KeyError as error:
            raise ValueError(f"Unsupported Qwen2 checkpoint dtype: {name}") from error

    @staticmethod
    def _torch_dtype(dtype: torch.dtype) -> DataType:
        dtypes = {
            torch.bfloat16: DataType.BF16,
            torch.float16: DataType.F16,
            torch.float32: DataType.F32,
        }
        try:
            return dtypes[dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported Qwen2 tensor dtype: {dtype}") from error

    def _load_weight(self, name: str, tensor: torch.Tensor) -> None:
        shape = (c_size_t * tensor.ndim)(*tensor.shape)
        status = Qwen2WeightLoadStatus(
            LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                self._model,
                name.encode("utf-8"),
                c_void_p(tensor.data_ptr()),
                int(self._torch_dtype(tensor.dtype)),
                shape,
                tensor.ndim,
            )
        )
        if status != Qwen2WeightLoadStatus.SUCCESS:
            raise ValueError(
                f"Failed to load Qwen2 weight {name} with shape {tuple(tensor.shape)} "
                f"and dtype {tensor.dtype}: {status.name}"
            )

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        tokens = [int(token) for token in inputs]
        if not tokens:
            raise ValueError("inputs must contain at least one token")
        if max_new_tokens is None:
            max_new_tokens = self._meta.maxseq - len(tokens)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if top_k != 1:
            raise ValueError("LLAISYS currently supports greedy decoding only (top_k=1)")
        if len(tokens) + max_new_tokens > self._meta.maxseq:
            raise ValueError("requested sequence exceeds max_position_embeddings")

        # Greedy decoding is deterministic, so top_p and temperature do not
        # affect the selected token when top_k is one.
        del top_p, temperature
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        output = list(tokens)
        step_input = tokens
        for _ in range(max_new_tokens):
            token_array = (c_int64 * len(step_input))(*step_input)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_array, len(step_input)
                )
            )
            output.append(next_token)
            if next_token == self._meta.end_token:
                break
            step_input = [next_token]

        return output
