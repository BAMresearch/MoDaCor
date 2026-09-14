# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import monotonic, sleep
from typing import IO, Any

from .runtime import RuntimeClient

__all__ = ["LocalRuntimeServer"]


class LocalRuntimeServer:
    """Manage a notebook-local runtime process without owning external servers."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_path: str | Path | None = None,
        environment: Mapping[str, str] | None = None,
        read_roots: Sequence[str | Path] = (),
        write_roots: Sequence[str | Path] = (),
        runtime_policy: str = "trusted",
        startup_timeout: float = 45.0,
        shutdown_timeout: float = 10.0,
        request_timeout: float = 120.0,
        executable: str | Path | None = None,
        max_sessions: int | None = None,
        max_pipeline_yaml_bytes: int | None = None,
        max_buffer_upload_bytes: int | None = None,
    ) -> None:
        if runtime_policy not in {"trusted", "restricted"}:
            raise ValueError("runtime_policy must be 'trusted' or 'restricted'.")
        if not 1 <= int(port) <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if startup_timeout <= 0 or shutdown_timeout <= 0 or request_timeout <= 0:
            raise ValueError("Runtime server timeouts must be positive.")
        self.host = str(host)
        self.port = int(port)
        self.log_path = None if log_path is None else Path(log_path)
        self.environment = dict(environment or {})
        self.read_roots = tuple(Path(path) for path in read_roots)
        self.write_roots = tuple(Path(path) for path in write_roots)
        self.runtime_policy = runtime_policy
        self.startup_timeout = float(startup_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self.executable = str(executable or sys.executable)
        self.max_sessions = max_sessions
        self.max_pipeline_yaml_bytes = max_pipeline_yaml_bytes
        self.max_buffer_upload_bytes = max_buffer_upload_bytes
        self.client = RuntimeClient(f"http://{self.host}:{self.port}", timeout=request_timeout)
        self.process: subprocess.Popen[str] | None = None
        self._log_file: IO[str] | None = None

    @property
    def launched(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self) -> RuntimeClient:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> RuntimeClient:
        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self._close_log()
        if self.client.is_ready():
            return self.client
        if self.launched:
            try:
                self.client.wait_until_ready(timeout=self.startup_timeout)
            except Exception:
                self.stop()
                raise
            return self.client

        self.process = None
        self._close_log()
        target = self._open_log()
        environment = os.environ.copy()
        environment.update(self.environment)
        try:
            self.process = subprocess.Popen(
                self._command(),
                stdout=target,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
            self._wait_for_startup()
        except Exception:
            self.stop()
            raise
        return self.client

    def _command(self) -> list[str]:
        command = [
            self.executable,
            "-m",
            "modacor.cli",
            "serve",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--runtime-policy",
            self.runtime_policy,
        ]
        for root in self.read_roots:
            command.extend(["--read-root", str(root)])
        for root in self.write_roots:
            command.extend(["--write-root", str(root)])
        for option, value in (
            ("--max-sessions", self.max_sessions),
            ("--max-pipeline-yaml-bytes", self.max_pipeline_yaml_bytes),
            ("--max-buffer-upload-bytes", self.max_buffer_upload_bytes),
        ):
            if value is not None:
                command.extend([option, str(value)])
        return command

    def _open_log(self) -> int | IO[str]:
        if self.log_path is None:
            return subprocess.DEVNULL
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        return self._log_file

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _wait_for_startup(self) -> None:
        deadline = monotonic() + self.startup_timeout
        while monotonic() < deadline:
            if self.client.is_ready():
                return
            assert self.process is not None
            if self.process.poll() is not None:
                log_note = f" See {self.log_path}." if self.log_path is not None else ""
                raise RuntimeError(
                    f"MoDaCor runtime exited during startup with code {self.process.returncode}.{log_note}"
                )
            sleep(min(0.25, max(0.0, deadline - monotonic())))
        raise TimeoutError(
            f"MoDaCor runtime did not become ready within {self.startup_timeout:g} seconds at {self.client.base_url}."
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.shutdown_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.shutdown_timeout)
        finally:
            self._close_log()
