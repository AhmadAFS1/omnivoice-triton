"""Tests for the FastAPI server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from omnivoice_triton.cli.api_server import create_app


class _FakeModel:
    def __init__(self) -> None:
        self.sampling_rate = 24000
        self._asr_pipe = None
        self.calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []

    def load_asr_model(self) -> None:
        self._asr_pipe = object()

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: str,
        ref_text: str | None,
        preprocess_prompt: bool,
    ) -> dict[str, Any]:
        call = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "preprocess_prompt": preprocess_prompt,
        }
        self.prompt_calls.append(call)
        return {"prompt": call}

    def generate(self, **kwargs: Any) -> list[np.ndarray]:
        self.calls.append(kwargs)
        return [np.zeros(2400, dtype=np.float32)]


class _FakeRunner:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.loaded = False
        self.unloaded = False
        self._model = _FakeModel()

    def load_model(self) -> None:
        self.loaded = True

    @property
    def model(self) -> _FakeModel:
        return self._model

    def unload_model(self) -> None:
        self.unloaded = True


def _make_app(tmp_path: Path, created: list[_FakeRunner], **kwargs: Any):
    def _factory(name: str, **factory_kwargs: Any) -> _FakeRunner:
        runner = _FakeRunner(name, **factory_kwargs)
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
    saved_path = Path(response.headers["x-omnivoice-saved-path"])
    assert saved_path.exists()

    model = created[0].model
    call = model.calls[0]
    assert call["text"] == "Hello from the API."
    assert call["instruct"] == "female, young adult"
    assert call["duration"] == 1.2
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
            files={"ref_audio": ("ref.wav", b"fake-wav-data", "audio/wav")},
        )

    assert response.status_code == 200
    model = created[0].model
    prompt_call = model.prompt_calls[0]
    assert prompt_call["ref_text"] == "Reference transcript."
    assert prompt_call["preprocess_prompt"] is False

    generate_call = model.calls[0]
    assert generate_call["instruct"] == "whisper"
    assert "voice_clone_prompt" in generate_call


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
