"""Hybrid OmniVoice runner: Triton kernel fusion + CUDA Graph."""

import logging

from omnivoice_triton.models.faster_runner import FasterRunner
from omnivoice_triton.models.patching import (
    apply_sage_attention,
    apply_triton_kernels,
    find_patchable_model,
)

logger = logging.getLogger(__name__)


class TritonFasterRunner(FasterRunner):
    """FasterRunner with Triton kernel patching applied BEFORE graph capture.

    Combines:
      1. Triton kernel fusion (RMSNorm, SwiGLU, Fused Norm+Residual)
      2. Optional SageAttention
      3. CUDA Graph capture and replay

    Triton patches (and SageAttention) are applied first so that the
    captured graph includes the optimized kernels.

    Args:
        patch_range: Half-open ``(start, end)`` range of decoder layer
            indices to patch. Defaults to ``(0, 24)``. ``None`` patches all.
        enable_sage_attention: Replace SDPA with SageAttention. Requires
            ``pip install sageattention``. Gracefully skips if unavailable.
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID.
        dtype: Model dtype string.
    """

    _supports_fused_rope = True

    def __init__(
        self,
        patch_range: tuple[int, int] | None = (0, 24),
        enable_sage_attention: bool = False,
        device: str = "cuda",
        model_id: str = "k2-fsa/OmniVoice",
        dtype: str = "fp16",
        decode_postprocess_workers: int = 0,
        max_cuda_graphs: int = 32,
        audio_decode_batch_size: int = 1,
        disable_unused_kv_cache: bool = False,
        split_cfg: bool = False,
        max_cuda_graph_row_tokens: int = 0,
        cuda_graph_capture_after_calls: int = 0,
        fused_rope: bool = False,
    ) -> None:
        super().__init__(
            device=device,
            model_id=model_id,
            dtype=dtype,
            decode_postprocess_workers=decode_postprocess_workers,
            max_cuda_graphs=max_cuda_graphs,
            audio_decode_batch_size=audio_decode_batch_size,
            disable_unused_kv_cache=disable_unused_kv_cache,
            split_cfg=split_cfg,
            max_cuda_graph_row_tokens=max_cuda_graph_row_tokens,
            cuda_graph_capture_after_calls=cuda_graph_capture_after_calls,
            fused_rope=fused_rope,
        )
        if fused_rope and (patch_range != (0, 24) or enable_sage_attention):
            raise ValueError(
                "Fused RoPE requires default (0,24) Triton patches and SDPA."
            )
        self.patch_range = patch_range
        self.enable_sage_attention = enable_sage_attention

    def load_model(self) -> None:
        """Load model, apply Triton patches, then install CUDA Graph wrapper."""
        with self._fused_rope_startup():
            self._load_hybrid_model()

    def _load_hybrid_model(self) -> None:
        # Load base model (without CUDA Graph wrapper yet)
        from omnivoice_triton.models.base_runner import BaseRunner

        BaseRunner.load_model(self)

        # Apply Triton kernel patches FIRST
        patchable = find_patchable_model(self._model)
        apply_triton_kernels(patchable, patch_range=self.patch_range)

        # Apply SageAttention if enabled
        if self.enable_sage_attention:
            apply_sage_attention(patchable, patch_range=self.patch_range)

        # Then install CUDA Graph wrapper on the patched model
        from omnivoice_triton.models.faster_runner import _CUDAGraphForward

        self._graph_forward = _CUDAGraphForward(
            self._model,
            max_graphs=self.max_cuda_graphs,
        )
        self._model.forward = self._graph_forward
        self._install_graph_memory_policy()
        self._install_split_cfg()
        self._install_fused_rope()

        logger.info(
            "HybridRunner ready (Triton kernels + CUDA Graph wrapper installed, "
            "max_graphs=%d).",
            self.max_cuda_graphs,
        )
