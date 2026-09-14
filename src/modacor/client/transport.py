# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
from urllib import error, request

from .errors import RuntimeAPIError

__all__ = ["RuntimeTransport"]


class RuntimeTransport(Protocol):
    """Transport contract accepted by :class:`RuntimeClient`."""

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        raise NotImplementedError


class UrllibRuntimeTransport:
    """Standard-library HTTP transport used by default."""

    def request(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        http_request = request.Request(url, method=method.upper(), data=data, headers=dict(headers))
        try:
            with request.urlopen(http_request, timeout=timeout) as response:  # noqa: S310
                return response.status, response.read(), dict(response.headers.items())
        except error.HTTPError as exc:
            response_headers = {} if exc.headers is None else dict(exc.headers.items())
            return exc.code, exc.read(), response_headers
        except (error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeAPIError(
                status=None,
                code="TRANSPORT_ERROR",
                message=str(reason),
                endpoint=url,
                method=method,
            ) from exc
