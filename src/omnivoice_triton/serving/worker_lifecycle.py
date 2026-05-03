"""Lingua worker callback lifecycle for Vast.ai OmniVoice workers."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_worker_type(value: str | None) -> str:
    worker_type = (value or "tts").strip().lower()
    if worker_type == "chatterbox":
        return "tts"
    return worker_type or "tts"


def _parse_positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(str(value).strip()) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(value: str | None, default: float) -> float:
    try:
        parsed = float(str(value).strip()) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0.0 else default


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _has_env_with_prefix(prefix: str) -> bool:
    return any(
        name.startswith(prefix) and value.strip() for name, value in os.environ.items()
    )


def _detect_instance_id() -> str:
    value = _env_first(
        "LINGUA_WORKER_INSTANCE_ID",
        "VAST_INSTANCE_ID",
        "INSTANCE_ID",
        "CONTAINER_ID",
        "HOSTNAME",
    )
    if value:
        return value
    return socket.gethostname()


@dataclass(frozen=True)
class WorkerPublicEndpoint:
    base_url: str | None
    public_ip: str | None
    public_port: str | None
    internal_port: int


def detect_public_endpoint(port: int) -> WorkerPublicEndpoint:
    """Detect the public endpoint Lingua should use to call this worker."""
    explicit = _env_first(
        "LINGUA_WORKER_BASE_URL",
        "LINGUA_WORKER_PUBLIC_BASE_URL",
        "OMNIVOICE_PUBLIC_BASE_URL",
        "PUBLIC_BASE_URL",
    )
    if explicit:
        return WorkerPublicEndpoint(
            base_url=explicit.rstrip("/"),
            public_ip=None,
            public_port=None,
            internal_port=port,
        )

    public_ip = _env_first(
        "LINGUA_WORKER_PUBLIC_IP",
        "VAST_PUBLIC_IP",
        "PUBLIC_IPADDR",
        "PUBLIC_IP",
    )
    public_port = _env_first(
        "LINGUA_WORKER_PUBLIC_PORT",
        "VAST_PUBLIC_PORT",
        f"VAST_TCP_PORT_{port}",
        "PUBLIC_PORT",
    )
    base_url: str | None = None
    if public_ip and public_port:
        base_url = f"http://{public_ip}:{public_port}"
    elif public_ip and not _has_env_with_prefix("VAST_TCP_PORT_"):
        base_url = f"http://{public_ip}:{port}"

    return WorkerPublicEndpoint(
        base_url=base_url,
        public_ip=public_ip,
        public_port=public_port,
        internal_port=port,
    )


def detect_public_base_url(port: int) -> str | None:
    """Detect the public URL Lingua should use to call this worker."""
    return detect_public_endpoint(port).base_url


class WorkerRuntimeState:
    """Thread-safe request and drain state shared by API and heartbeat loop."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._active_requests = 0
        self._draining = False

    def begin_request(self) -> bool:
        with self._lock:
            if self._draining:
                return False
            self._active_requests += 1
            return True

    def end_request(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def request_drain(self) -> None:
        with self._lock:
            self._draining = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_requests = self._active_requests
            draining = self._draining
        return {
            "status": "draining" if draining else "healthy",
            "active_requests": active_requests,
            "queue_depth": active_requests,
            "capacity": self.capacity,
            "assignable": not draining and active_requests < self.capacity,
            "draining": draining,
        }


@dataclass(frozen=True)
class WorkerCallbackConfig:
    worker_type: str
    worker_id: str
    instance_id: str
    register_url: str
    heartbeat_url: str
    token: str
    capacity: int
    public_base_url: str | None
    public_ip: str | None
    public_port: str | None
    internal_port: int
    region: str | None
    gpu_type: str | None
    heartbeat_interval_s: float
    request_timeout_s: float

    @classmethod
    def from_env(cls, *, port: int) -> "WorkerCallbackConfig | None":
        token = _env_first("LINGUA_WORKER_TOKEN")
        base_url = _env_first("LINGUA_CONTROL_PLANE_BASE_URL")
        register_url = _env_first("LINGUA_WORKER_REGISTER_URL")
        heartbeat_url = _env_first("LINGUA_WORKER_HEARTBEAT_URL")
        if not token or not (base_url or (register_url and heartbeat_url)):
            return None

        if base_url:
            base_url = base_url.rstrip("/")
        register_url = register_url or f"{base_url}/api/runtime/workers/register"
        heartbeat_url = heartbeat_url or f"{base_url}/api/runtime/workers/heartbeat"
        worker_type = _normalize_worker_type(_env_first("LINGUA_WORKER_TYPE"))
        instance_id = _detect_instance_id()
        worker_id = _env_first("LINGUA_WORKER_ID") or f"{worker_type}-{instance_id}"
        capacity = _parse_positive_int(
            _env_first(
                "LINGUA_WORKER_DEFAULT_CAPACITY",
                "OMNIVOICE_WORKER_DEFAULT_CAPACITY",
            ),
            1,
        )
        public_endpoint = detect_public_endpoint(port)
        return cls(
            worker_type=worker_type,
            worker_id=worker_id,
            instance_id=instance_id,
            register_url=register_url,
            heartbeat_url=heartbeat_url,
            token=token,
            capacity=capacity,
            public_base_url=public_endpoint.base_url,
            public_ip=public_endpoint.public_ip,
            public_port=public_endpoint.public_port,
            internal_port=public_endpoint.internal_port,
            region=_env_first("LINGUA_WORKER_REGION"),
            gpu_type=_env_first("LINGUA_WORKER_GPU_TYPE"),
            heartbeat_interval_s=_parse_positive_float(
                _env_first("LINGUA_WORKER_HEARTBEAT_INTERVAL_SECONDS"),
                20.0,
            ),
            request_timeout_s=_parse_positive_float(
                _env_first("LINGUA_WORKER_CALLBACK_TIMEOUT_SECONDS"),
                8.0,
            ),
        )


class WorkerLifecycleReporter:
    """Registers this worker with Lingua and sends periodic heartbeats."""

    def __init__(
        self,
        config: WorkerCallbackConfig,
        runtime_state: WorkerRuntimeState,
    ) -> None:
        self.config = config
        self.runtime_state = runtime_state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="lingua-worker-callback",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def payload(self) -> dict[str, Any]:
        snapshot = self.runtime_state.snapshot()
        metadata = {
            "provider": "omnivoice",
            "model_id": "omnivoice",
            "queue_depth": snapshot["queue_depth"],
            "active_requests": snapshot["active_requests"],
            "max_concurrent": snapshot["capacity"],
        }
        return {
            "worker_type": self.config.worker_type,
            "worker_id": self.config.worker_id,
            "instance_id": self.config.instance_id,
            "base_url": self.config.public_base_url,
            "endpoint_url": self.config.public_base_url,
            "public_ip": self.config.public_ip,
            "public_port": self.config.public_port,
            "internal_port": self.config.internal_port,
            "status": snapshot["status"],
            "capacity": snapshot["capacity"],
            "region": self.config.region,
            "gpu_type": self.config.gpu_type,
            "metadata": metadata,
        }

    def _run(self) -> None:
        if not self.config.public_base_url:
            logger.warning(
                "Lingua worker callback disabled: public base URL could not be "
                "detected. "
                "Set LINGUA_WORKER_PUBLIC_BASE_URL for Vast.ai registration."
            )
            return

        self._retry_until_registered()
        while not self._stop_event.wait(self.config.heartbeat_interval_s):
            self._post_with_backoff(
                self.config.heartbeat_url,
                "heartbeat",
                forever=False,
            )

    def _retry_until_registered(self) -> None:
        backoff_s = 1.0
        while not self._stop_event.is_set():
            if self._post(self.config.register_url, "register"):
                return
            self._stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)

    def _post_with_backoff(self, url: str, action: str, *, forever: bool) -> bool:
        backoff_s = 1.0
        while not self._stop_event.is_set():
            if self._post(url, action):
                return True
            if not forever:
                return False
            self._stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)
        return False

    def _post(self, url: str, action: str) -> bool:
        body = json.dumps(self.payload()).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Worker-Token": self.config.token,
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_s,
            ) as response:
                if 200 <= int(response.status) < 300:
                    logger.info(
                        "Lingua worker %s succeeded worker_id=%s status=%s",
                        action,
                        self.config.worker_id,
                        self.runtime_state.snapshot()["status"],
                    )
                    return True
                logger.warning(
                    "Lingua worker %s failed worker_id=%s status_code=%s",
                    action,
                    self.config.worker_id,
                    response.status,
                )
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Lingua worker %s failed worker_id=%s status_code=%s",
                action,
                self.config.worker_id,
                exc.code,
            )
        except Exception as exc:
            logger.warning(
                "Lingua worker %s failed worker_id=%s error=%s",
                action,
                self.config.worker_id,
                exc,
            )
        return False
