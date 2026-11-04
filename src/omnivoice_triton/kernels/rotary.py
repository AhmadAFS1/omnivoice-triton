"""Inference-only FP16 Qwen3 rotary kernel with eager intermediate rounding.

The two half products are separately rounded before addition. Runtime sequence
lengths and strides avoid per-language JIT specialization. TensorIterator meta
semantics preserve the original output strides for downstream attention.
"""

import functools

import torch
import triton
import triton.language as tl

RUNTIME_SCALARS = [
    "S",
    "N",
    "X_B",
    "X_H",
    "X_S",
    "X_D",
    "C_B",
    "C_S",
    "C_D",
    "T_B",
    "T_S",
    "T_D",
    "Y_B",
    "Y_H",
    "Y_S",
    "Y_D",
]


@triton.jit(
    do_not_specialize=RUNTIME_SCALARS,
    do_not_specialize_on_alignment=[*RUNTIME_SCALARS, "X", "COS", "SIN", "OUT"],
)
def _rounded_rope_kernel(
    X,
    COS,
    SIN,
    OUT,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    N,
    X_B,
    X_H,
    X_S,
    X_D,
    C_B,
    C_S,
    C_D,
    T_B,
    T_S,
    T_D,
    Y_B,
    Y_H,
    Y_S,
    Y_D,
    BLOCK: tl.constexpr,
):
    head_column = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    valid = (head_column < H * D) & (row * H * D + head_column < N)
    column = head_column % D
    head = head_column // D
    sequence = row % S
    batch = row // S
    base = batch * X_B + head * X_H + sequence * X_S
    x = tl.load(X + base + column * X_D, valid, other=0).to(tl.float32)
    other_column = (column + D // 2) % D
    other = tl.load(X + base + other_column * X_D, valid, other=0).to(tl.float32)
    rotated = tl.where(column < D // 2, -other, other)
    cosine = tl.load(
        COS + batch * C_B + sequence * C_S + column * C_D, valid, other=0
    ).to(tl.float32)
    sine = tl.load(
        SIN + batch * T_B + sequence * T_S + column * T_D, valid, other=0
    ).to(tl.float32)
    # These explicit low-precision roundings match eager PyTorch's two
    # independent half TensorIterator multiply results before their half add.
    left = (x * cosine).to(tl.float16)
    right = (rotated * sine).to(tl.float16)
    output = (left.to(tl.float32) + right.to(tl.float32)).to(tl.float16)
    tl.store(
        OUT + batch * Y_B + head * Y_H + sequence * Y_S + column * Y_D,
        output,
        valid,
    )


def reference_rope(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)

    def one(x):
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (x * cos) + (rotated * sin)

    return one(q), one(k)


@functools.lru_cache(maxsize=128)
def output_layout(
    q_shape, q_stride, k_shape, k_stride, c_shape, c_stride, t_shape, t_stride
):
    """Use actual Torch meta semantics for broadcast/output layout selection."""
    tensors = [
        torch.empty_strided(shape, stride, dtype=torch.float16, device="meta")
        for shape, stride in [
            (q_shape, q_stride),
            (k_shape, k_stride),
            (c_shape, c_stride),
            (t_shape, t_stride),
        ]
    ]
    q_out, k_out = reference_rope(*tensors)
    return tuple(q_out.stride()), tuple(k_out.stride())


def unsupported_reason(q, k, cos, sin, unsqueeze_dim=1):
    tensors = [q, k, cos, sin]
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        return "non-tensor input"
    if (
        unsqueeze_dim != 1
        or q.ndim != 4
        or k.ndim != 4
        or cos.ndim != 3
        or sin.ndim != 3
    ):
        return "unsupported rank or broadcast dimension"
    if any(x.device.type != "cuda" or x.device != q.device for x in tensors):
        return "requires one CUDA device"
    if any(x.dtype != torch.float16 for x in tensors):
        return "prototype supports FP16 only"
    if any(x.requires_grad for x in tensors) or torch.is_grad_enabled():
        return "inference-only prototype"
    if torch.is_autocast_enabled("cuda"):
        return "autocast unsupported"
    batch, _, sequence, dim = q.shape
    reason = geometry_reason(tuple(q.shape), tuple(k.shape))
    if reason:
        return reason
    if dim != 128 or q.shape[1] != 16 or k.shape[1] != 8:
        return "pinned Qwen3 head geometry required"
    if k.shape[0] != batch or k.shape[2:] != (sequence, dim):
        return "Q/K shape mismatch"
    for value in [cos, sin]:
        if value.shape[0] not in [1, batch] or value.shape[1:] != (sequence, dim):
            return "unsupported cosine/sine broadcast shape"
    if any(x.numel() == 0 or any(stride < 0 for stride in x.stride()) for x in tensors):
        return "empty input or negative stride"
    for value in tensors:
        if any(stride > 2**31 - 1 for stride in value.stride()):
            return "stride exceeds pinned int32 scalar bound"
        maximum_offset = sum(
            (size - 1) * stride
            for size, stride in zip(value.shape, value.stride(), strict=True)
        )
        if maximum_offset > 2**31 - 1:
            return "storage offset exceeds pinned int32 indexing bound"
    return ""


def geometry_reason(q_shape, k_shape):
    if len(q_shape) != 4 or len(k_shape) != 4:
        return "rank4 required"
    batch, heads, sequence, dim = q_shape
    if not 1 <= batch <= 16 or not 1 <= sequence <= 4096:
        return "outside pinned B1..16 and S1..4096 probe allowlist"
    if batch * sequence > 65535:
        return "2D CUDA grid Y exceeds65535; fallback required before launch"
    if heads != 16 or dim != 128 or k_shape != (batch, 8, sequence, dim):
        return "pinned Qwen3 head geometry required"
    if batch * heads * sequence * dim > 2**31 - 1:
        return "element count exceeds pinned int32 bound"
    return ""


def triton_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply to qualified tensors; runner startup must warm both head shapes."""
    reason = unsupported_reason(q, k, cos, sin, unsqueeze_dim)
    if reason:
        raise ValueError(reason)
    with torch.cuda.device(q.device):
        return _apply_rotary(q, k, cos, sin)


def _apply_rotary(q, k, cos, sin):
    """Qualified owner-only implementation; caller has checked tensor metadata."""
    q_stride, k_stride = output_layout(
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.shape),
        tuple(k.stride()),
        tuple(cos.shape),
        tuple(cos.stride()),
        tuple(sin.shape),
        tuple(sin.stride()),
    )
    q_out = torch.empty_strided(q.shape, q_stride, dtype=q.dtype, device=q.device)
    k_out = torch.empty_strided(k.shape, k_stride, dtype=k.dtype, device=k.device)
    for value, result in [(q, q_out), (k, k_out)]:
        batch, heads, sequence, dim = value.shape
        cosine_stride = (0 if cos.shape[0] == 1 else cos.stride(0), *cos.stride()[1:])
        sine_stride = (0 if sin.shape[0] == 1 else sin.stride(0), *sin.stride()[1:])
        arguments = (
            value,
            cos,
            sin,
            result,
            heads,
            sequence,
            dim,
            batch * heads * sequence * dim,
            *value.stride(),
            *cosine_stride,
            *sine_stride,
            *result.stride(),
        )
        options = {"BLOCK": 512, "num_warps": 4, "enable_fp_fusion": False}
        grid = (triton.cdiv(heads * dim, 512), batch * sequence)
        _rounded_rope_kernel[grid](*arguments, **options)
    return q_out, k_out
