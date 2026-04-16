"""FastAPI server for omnivoice-triton runners."""

from __future__ import annotations

import argparse
import io
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

from omnivoice_triton import ALL_RUNNER_NAMES, __version__, create_runner

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_PORT = 8002
DEFAULT_RUNNER = "hybrid"
DEFAULT_SAMPLE_RATE = 24000
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class GenerateMode(str, Enum):
    """Supported request modes."""

    auto = "auto"
    design = "design"
    clone = "clone"


def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Launch a FastAPI server for omnivoice-triton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        choices=ALL_RUNNER_NAMES,
        help="Runner backend to use.",
    )
    parser.add_argument(
        "--sage-attention",
        action="store_true",
        default=False,
        help="Enable SageAttention for Triton/Hybrid runners.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["bf16", "fp16", "fp32"],
        help="Runner dtype.",
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP to bind to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy deployments.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR at startup. Clone mode without ref_text "
        "may still load ASR on demand.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to persist a copy of each generated WAV.",
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_language(language: str | None) -> str | None:
    value = _normalize_text(language)
    if value is None:
        return None
    if value.lower() in {"auto", "none"}:
        return None
    return value


def _parse_positive_int(name: str, raw: str | None, default: int) -> int:
    value = _normalize_text(raw)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise HTTPException(400, detail=f"{name} must be > 0.")
    return parsed


def _parse_nonnegative_float(name: str, raw: str | None, default: float) -> float:
    value = _normalize_text(raw)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be a number.") from exc
    if parsed < 0.0:
        raise HTTPException(400, detail=f"{name} must be >= 0.")
    return parsed


def _parse_optional_positive_float(name: str, raw: str | None) -> float | None:
    value = _normalize_text(raw)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be a number.") from exc
    if parsed <= 0.0:
        raise HTTPException(400, detail=f"{name} must be > 0.")
    return parsed


def _parse_optional_duration(raw: str | None) -> float | None:
    """Parse duration the same way as the Gradio demo.

    The upstream demo treats empty or non-positive duration as "unset" and
    falls back to speed / duration estimation instead of raising an error.
    """
    value = _normalize_text(raw)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail="duration must be a number.") from exc
    if parsed <= 0.0:
        return None
    return parsed


def _reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def _to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().float().numpy()
    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return np.clip(waveform, -1.0, 1.0)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not slug:
        slug = "tts"
    return slug[:max_len].strip("-") or "tts"


def _save_output(
    wav_bytes: bytes,
    save_dir: Path,
    mode: GenerateMode,
    text: str,
    request_id: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_path = save_dir / f"{stamp}_{mode.value}_{_slugify(text)}_{request_id}.wav"
    output_path.write_bytes(wav_bytes)
    return output_path


def _build_runner(
    runner_name: str,
    *,
    model_checkpoint: str,
    device: str,
    dtype: str,
    enable_sage_attention: bool,
    runner_factory: Callable[..., Any],
) -> Any:
    kwargs: dict[str, Any] = {
        "device": device,
        "model_id": model_checkpoint,
        "dtype": dtype,
    }
    if enable_sage_attention:
        if runner_name not in {"triton", "hybrid"}:
            raise ValueError(
                "--sage-attention is only supported for the triton and hybrid runners."
            )
        kwargs["enable_sage_attention"] = True
    return runner_factory(runner_name, **kwargs)


def create_app(
    model_checkpoint: str = DEFAULT_MODEL_ID,
    runner_name: str = DEFAULT_RUNNER,
    device: str | None = None,
    dtype: str = "fp16",
    load_asr: bool = True,
    enable_sage_attention: bool = False,
    save_dir: str | None = None,
    runner_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """Create the FastAPI app."""
    resolved_device = device or get_best_device()
    if not resolved_device.startswith("cuda") and runner_name != "base":
        raise ValueError(
            f"Runner '{runner_name}' requires CUDA. Use --runner base or a CUDA device."
        )

    factory = runner_factory or create_runner

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner = _build_runner(
            runner_name,
            model_checkpoint=model_checkpoint,
            device=resolved_device,
            dtype=dtype,
            enable_sage_attention=enable_sage_attention,
            runner_factory=factory,
        )
        runner.load_model()
        model = runner.model
        if load_asr and hasattr(model, "load_asr_model"):
            model.load_asr_model()
        app.state.runner = runner
        app.state.sample_rate = int(
            getattr(model, "sampling_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE
        )
        logger.info(
            "API server ready runner=%s model=%s device=%s dtype=%s asr_loaded=%s",
            runner_name,
            model_checkpoint,
            resolved_device,
            dtype,
            bool(getattr(model, "_asr_pipe", None)),
        )
        try:
            yield
        finally:
            if app.state.runner is not None:
                app.state.runner.unload_model()
                app.state.runner = None

    app = FastAPI(
        title="omnivoice-triton API",
        version=__version__,
        description="HTTP API for OmniVoice text-to-speech with Triton runners.",
        lifespan=lifespan,
    )
    app.state.model_checkpoint = model_checkpoint
    app.state.runner_name = runner_name
    app.state.device = resolved_device
    app.state.dtype = dtype
    app.state.load_asr = load_asr
    app.state.enable_sage_attention = enable_sage_attention
    app.state.save_dir = Path(save_dir) if save_dir else None
    app.state.runner = None
    app.state.sample_rate = DEFAULT_SAMPLE_RATE
    app.state.generate_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        runner = app.state.runner
        model = getattr(runner, "model", None)
        return {
            "status": "ok",
            "model": app.state.model_checkpoint,
            "runner": app.state.runner_name,
            "device": app.state.device,
            "dtype": app.state.dtype,
            "sample_rate": app.state.sample_rate,
            "load_asr": app.state.load_asr,
            "asr_loaded": bool(getattr(model, "_asr_pipe", None)),
            "save_dir": str(app.state.save_dir) if app.state.save_dir else None,
            "sage_attention": bool(app.state.enable_sage_attention),
        }

    @app.get("/languages")
    def languages() -> dict[str, Any]:
        items = [
            {
                "id": LANG_NAME_TO_ID[name],
                "name": name,
                "display_name": lang_display_name(name),
            }
            for name in sorted(LANG_NAME_TO_ID, key=lang_display_name)
        ]
        return {"count": len(items), "languages": items}

    @app.post(
        "/generate",
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}}}},
    )
    def generate(
        mode: GenerateMode = Form(...),
        text: str = Form(...),
        language: str | None = Form(None),
        instruct: str | None = Form(None),
        ref_text: str | None = Form(None),
        num_step: str | None = Form(None),
        guidance_scale: str | None = Form(None),
        t_shift: str | None = Form(None),
        layer_penalty_factor: str | None = Form(None),
        position_temperature: str | None = Form(None),
        class_temperature: str | None = Form(None),
        speed: str | None = Form(None),
        duration: str | None = Form(None),
        denoise: bool = Form(True),
        preprocess_prompt: bool = Form(True),
        postprocess_output: bool = Form(True),
        ref_audio: UploadFile | None = File(None),
    ) -> Response:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, detail="Model is not ready yet.")

        text_value = _normalize_text(text)
        if text_value is None:
            raise HTTPException(400, detail="text is required.")

        instruct_value = _normalize_text(instruct)
        ref_text_value = _normalize_text(ref_text)
        language_value = _normalize_language(language)

        if mode == GenerateMode.auto:
            if instruct_value is not None:
                raise HTTPException(
                    400, detail="instruct is not allowed for mode=auto."
                )
            if ref_audio is not None or ref_text_value is not None:
                raise HTTPException(
                    400, detail="ref_audio/ref_text are not allowed for mode=auto."
                )
        elif mode == GenerateMode.design:
            if ref_audio is not None or ref_text_value is not None:
                raise HTTPException(
                    400, detail="ref_audio/ref_text are not allowed for mode=design."
                )
        elif ref_audio is None:
            raise HTTPException(400, detail="ref_audio is required for mode=clone.")

        num_step_value = _parse_positive_int("num_step", num_step, 32)
        guidance_scale_value = _parse_nonnegative_float(
            "guidance_scale", guidance_scale, 2.0
        )
        t_shift_value = _parse_nonnegative_float("t_shift", t_shift, 0.1)
        layer_penalty_factor_value = _parse_nonnegative_float(
            "layer_penalty_factor", layer_penalty_factor, 5.0
        )
        position_temperature_value = _parse_nonnegative_float(
            "position_temperature", position_temperature, 5.0
        )
        class_temperature_value = _parse_nonnegative_float(
            "class_temperature", class_temperature, 0.0
        )
        speed_value = _parse_optional_positive_float("speed", speed)
        duration_value = _parse_optional_duration(duration)

        request_id = uuid4().hex[:12]
        started_at = _utc_now()
        ref_audio_path: Path | None = None
        saved_path: Path | None = None
        has_ref_audio = ref_audio is not None

        try:
            model = runner.model
            gen_config = OmniVoiceGenerationConfig(
                num_step=num_step_value,
                guidance_scale=guidance_scale_value,
                t_shift=t_shift_value,
                layer_penalty_factor=layer_penalty_factor_value,
                position_temperature=position_temperature_value,
                class_temperature=class_temperature_value,
                denoise=denoise,
                preprocess_prompt=preprocess_prompt,
                postprocess_output=postprocess_output,
            )

            generate_kwargs: dict[str, Any] = {
                "text": text_value,
                "generation_config": gen_config,
            }
            if language_value is not None:
                generate_kwargs["language"] = language_value
            if speed_value is not None and speed_value != 1.0:
                generate_kwargs["speed"] = speed_value
            if duration_value is not None:
                generate_kwargs["duration"] = duration_value

            with app.state.generate_lock:
                _reset_peak_vram()

                if mode == GenerateMode.design and instruct_value is not None:
                    generate_kwargs["instruct"] = instruct_value
                elif mode == GenerateMode.clone:
                    assert ref_audio is not None  # for type checkers
                    suffix = (
                        Path(ref_audio.filename or "reference.wav").suffix or ".wav"
                    )
                    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        ref_audio_path = Path(tmp.name)
                        tmp.write(ref_audio.file.read())
                    generate_kwargs["voice_clone_prompt"] = (
                        model.create_voice_clone_prompt(
                            ref_audio=str(ref_audio_path),
                            ref_text=ref_text_value,
                            preprocess_prompt=preprocess_prompt,
                        )
                    )
                    if instruct_value is not None:
                        generate_kwargs["instruct"] = instruct_value

                audio_output = model.generate(**generate_kwargs)

            audio = _to_numpy(audio_output)
            sample_rate = int(
                getattr(model, "sampling_rate", app.state.sample_rate)
                or app.state.sample_rate
            )
            wav_bytes = _to_wav_bytes(audio, sample_rate)
            finished_at = _utc_now()
            latency_ms = (finished_at - started_at).total_seconds() * 1000.0
            audio_duration_s = float(audio.shape[0]) / float(sample_rate)
            rtf = (
                latency_ms / 1000.0 / audio_duration_s if audio_duration_s > 0 else None
            )
            peak_vram_gb = _get_peak_vram_gb()

            if app.state.save_dir is not None:
                saved_path = _save_output(
                    wav_bytes=wav_bytes,
                    save_dir=app.state.save_dir,
                    mode=mode,
                    text=text_value,
                    request_id=request_id,
                )

            headers = {
                "Content-Disposition": 'inline; filename="omnivoice.wav"',
                "X-OmniVoice-Request-Id": request_id,
                "X-OmniVoice-Started-At": _iso_utc(started_at),
                "X-OmniVoice-Finished-At": _iso_utc(finished_at),
                "X-OmniVoice-Latency-Ms": f"{latency_ms:.2f}",
                "X-OmniVoice-Audio-Duration-S": f"{audio_duration_s:.3f}",
                "X-OmniVoice-Runner": app.state.runner_name,
                "X-OmniVoice-Device": app.state.device,
                "X-OmniVoice-Peak-Vram-Gb": f"{peak_vram_gb:.3f}",
            }
            if rtf is not None:
                headers["X-OmniVoice-RTF"] = f"{rtf:.4f}"
            if saved_path is not None:
                headers["X-OmniVoice-Saved-Path"] = str(saved_path)

            logger.info(
                "request_id=%s status=success mode=%s runner=%s started_at=%s "
                "finished_at=%s latency_ms=%.2f audio_s=%.3f rtf=%s text_chars=%d "
                "has_ref_audio=%s language=%s device=%s saved_path=%s",
                request_id,
                mode.value,
                app.state.runner_name,
                _iso_utc(started_at),
                _iso_utc(finished_at),
                latency_ms,
                audio_duration_s,
                f"{rtf:.4f}" if rtf is not None else "-",
                len(text_value),
                has_ref_audio,
                language_value or "auto",
                app.state.device,
                str(saved_path) if saved_path is not None else "-",
            )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("request_id=%s status=runtime_error", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive server guard
            logger.exception("request_id=%s status=unexpected_error", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if ref_audio is not None:
                ref_audio.file.close()
            if ref_audio_path is not None:
                ref_audio_path.unlink(missing_ok=True)

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(
            logging,
            os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        ),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = create_app(
        model_checkpoint=args.model,
        runner_name=args.runner,
        device=args.device,
        dtype=args.dtype,
        load_asr=not args.no_asr,
        enable_sage_attention=args.sage_attention,
        save_dir=args.save_dir,
    )

    uvicorn.run(
        app,
        host=args.ip,
        port=args.port,
        root_path=args.root_path or "",
    )


if __name__ == "__main__":
    main()
