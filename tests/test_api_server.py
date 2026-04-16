"""Tests for the FastAPI server."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from omnivoice_triton.cli.api_server import create_app


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
    ) -> SimpleNamespace:
        token_count = max(1, len(text.split()))
        return SimpleNamespace(
            input_ids=torch.ones((1, token_count), dtype=torch.long)
        )


def _fake_wav_bytes() -> bytes:
    buffer = BytesIO()
    sf.write(
        buffer,
        np.zeros(1600, dtype=np.float32),
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


class _FakeModel:
    def __init__(self, *, generate_delay_s: float = 0.0) -> None:
        self.sampling_rate = 24000
        self._asr_pipe = None
        self.text_tokenizer = _FakeTokenizer()
        self.audio_tokenizer = SimpleNamespace(
            config=SimpleNamespace(frame_rate=75),
        )
        self.calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []
        self.generate_delay_s = generate_delay_s

    def load_asr_model(self) -> None:
        self._asr_pipe = object()

    def _estimate_target_tokens(
        self,
        text: str,
        ref_text: str | None,
        num_ref_audio_tokens: int | None,
        *,
        speed: float = 1.0,
    ) -> int:
        del ref_text
        base = max(1, len(text.split()) * 12)
        if num_ref_audio_tokens is not None:
            base += min(num_ref_audio_tokens // 4, 20)
        return max(1, int(base / speed))

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: Any,
        ref_text: str | None,
        preprocess_prompt: bool,
    ) -> SimpleNamespace:
        if isinstance(ref_audio, tuple):
            waveform, sample_rate = ref_audio
            ref_audio_meta = {
                "shape": tuple(waveform.shape),
                "sample_rate": sample_rate,
            }
        else:
            ref_audio_meta = {"value": ref_audio}
        call = {
            "ref_audio": ref_audio_meta,
            "ref_text": ref_text,
            "preprocess_prompt": preprocess_prompt,
        }
        self.prompt_calls.append(call)
        return SimpleNamespace(
            ref_audio_tokens=torch.zeros((8, 48), dtype=torch.long),
            ref_text=ref_text or "Auto transcript.",
            ref_rms=0.2,
        )

    def generate(self, **kwargs: Any) -> list[np.ndarray]:
        self.calls.append(kwargs)
        texts = kwargs["text"]
        if isinstance(texts, str):
            texts = [texts]
        if self.generate_delay_s > 0:
            time.sleep(self.generate_delay_s)
        return [
            np.full(2400, 0.1 + i * 0.05, dtype=np.float32)
            for i, _ in enumerate(texts)
        ]


class _FakeRunner:
    def __init__(
        self,
        name: str,
        *,
        model_generate_delay_s: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.kwargs = kwargs
        self.loaded = False
        self.unloaded = False
        self._model = _FakeModel(generate_delay_s=model_generate_delay_s)

    def load_model(self) -> None:
        self.loaded = True

    @property
    def model(self) -> _FakeModel:
        return self._model

    def unload_model(self) -> None:
        self.unloaded = True


def _make_app(
    tmp_path: Path,
    created: list[_FakeRunner],
    *,
    model_generate_delay_s: float = 0.0,
    **kwargs: Any,
):
    def _factory(name: str, **factory_kwargs: Any) -> _FakeRunner:
        runner = _FakeRunner(
            name,
            model_generate_delay_s=model_generate_delay_s,
            **factory_kwargs,
        )
        created.append(runner)
        return runner

    return create_app(
        model_checkpoint="fake/omnivoice",
        runner_name="hybrid",
        device="cuda",
        dtype="fp16",
        runner_factory=_factory,
        save_dir=str(tmp_path),
        **kwargs,
    )


def test_health_and_languages(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=True, enable_sage_attention=True)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        payload = health.json()
        assert payload["status"] == "ok"
        assert payload["runner"] == "hybrid"
        assert payload["model"] == "fake/omnivoice"
        assert payload["device"] == "cuda"
        assert payload["asr_loaded"] is True
        assert payload["sage_attention"] is True
        assert payload["batch_collect_ms"] == 10.0
        assert payload["max_batch_requests"] == 32
        assert Path(payload["save_dir"]) == tmp_path

        languages = client.get("/languages")
        assert languages.status_code == 200
        lang_payload = languages.json()
        assert lang_payload["count"] > 0
        assert {"id", "name", "display_name"} <= set(lang_payload["languages"][0])

    assert created[0].loaded is True
    assert created[0].unloaded is True
    assert created[0].kwargs["enable_sage_attention"] is True


def test_generate_design_returns_wav_and_saves_file(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "design",
                "text": "Hello from the API.",
                "instruct": "female, young adult",
                "num_step": "16",
                "guidance_scale": "1.5",
                "position_temperature": "0.0",
                "class_temperature": "0.0",
                "duration": "1.2",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-omnivoice-latency-ms"]
    assert response.headers["x-omnivoice-audio-duration-s"]
    assert response.headers["x-omnivoice-batch-requests"] == "1"
    saved_path = Path(response.headers["x-omnivoice-saved-path"])
    assert saved_path.exists()

    model = created[0].model
    call = model.calls[0]
    assert call["text"] == ["Hello from the API."]
    assert call["instruct"] == ["female, young adult"]
    assert call["duration"] == [1.2]
    assert call["generation_config"].num_step == 16
    assert call["generation_config"].guidance_scale == 1.5


def test_generate_design_without_instruct_matches_gradio_fallback(
    tmp_path: Path,
) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "design",
                "text": "Fallback to auto-style generation.",
                "duration": "0",
            },
        )

    assert response.status_code == 200
    call = created[0].model.calls[0]
    assert "instruct" not in call
    assert "duration" not in call


def test_generate_clone_allows_instruct_and_prompt_options(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            data={
                "mode": "clone",
                "text": "Clone this voice.",
                "ref_text": "Reference transcript.",
                "instruct": "whisper",
                "preprocess_prompt": "false",
            },
            files={"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    model = created[0].model
    prompt_call = model.prompt_calls[0]
    assert prompt_call["ref_text"] == "Reference transcript."
    assert prompt_call["preprocess_prompt"] is False

    generate_call = model.calls[0]
    assert generate_call["instruct"] == ["whisper"]
    assert "voice_clone_prompt" in generate_call
    assert len(generate_call["voice_clone_prompt"]) == 1


def test_generate_clone_reuses_cached_prompt(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    request_kwargs = {
        "data": {
            "mode": "clone",
            "text": "Clone this voice again.",
            "ref_text": "Reference transcript.",
        },
        "files": {"ref_audio": ("ref.wav", _fake_wav_bytes(), "audio/wav")},
    }

    with TestClient(app) as client:
        first = client.post("/generate", **request_kwargs)
        second = client.post("/generate", **request_kwargs)

    assert first.status_code == 200
    assert second.status_code == 200
    model = created[0].model
    assert len(model.prompt_calls) == 1
    assert len(model.calls) == 2


def test_generate_batches_concurrent_requests(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(
        tmp_path,
        created,
        load_asr=False,
        batch_collect_ms=50.0,
        max_batch_requests=8,
        max_batch_target_tokens=10000,
        max_batch_conditioning_tokens=10000,
        max_batch_padding_ratio=10.0,
        model_generate_delay_s=0.05,
    )

    texts = [
        "Batch request number one.",
        "Batch request number two.",
        "Batch request number three.",
        "Batch request number four.",
    ]

    with TestClient(app) as client:
        start_barrier = threading.Barrier(len(texts))

        def _send(text: str):
            start_barrier.wait()
            return client.post(
                "/generate",
                data={"mode": "design", "text": text, "instruct": "warm"},
            )

        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [executor.submit(_send, text) for text in texts]
            responses = [future.result() for future in futures]

    assert all(response.status_code == 200 for response in responses)
    batch_sizes = [
        int(response.headers["x-omnivoice-batch-requests"]) for response in responses
    ]
    assert max(batch_sizes) >= 2

    model = created[0].model
    assert len(model.calls) < len(texts)
    flattened = [text for call in model.calls for text in call["text"]]
    assert sorted(flattened) == sorted(texts)


def test_generate_validation_errors(tmp_path: Path) -> None:
    created: list[_FakeRunner] = []
    app = _make_app(tmp_path, created, load_asr=False)

    with TestClient(app) as client:
        auto_with_instruct = client.post(
            "/generate",
            data={"mode": "auto", "text": "hello", "instruct": "female"},
        )
        assert auto_with_instruct.status_code == 400

        design_missing_instruct = client.post(
            "/generate",
            data={"mode": "design", "text": "hello"},
        )
        assert design_missing_instruct.status_code == 200

        clone_missing_audio = client.post(
            "/generate",
            data={"mode": "clone", "text": "hello"},
        )
        assert clone_missing_audio.status_code == 400
