"""Owner-loop cleanup and closed-loop recovery for auxiliary HTTP clients."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _close_orphan_stream(obj: Any, seen: set[int]) -> None:
    if obj is None or id(obj) in seen:
        return
    seen.add(id(obj))
    transport = getattr(obj, "_transport", None)
    sock = getattr(transport, "_sock", None)
    if sock is not None:
        sock.close()
    for attr in ("_stream", "transport_stream", "_connection", "_network_stream"):
        _close_orphan_stream(getattr(obj, attr, None), seen)


def close_orphaned_httpx(client: Any, owner_loop: Any) -> None:
    """Release FDs only after the owning loop can no longer perform TLS I/O."""
    if owner_loop is None or not owner_loop.is_closed():
        return
    try:
        import httpx
        from httpx._client import ClientState

        inner = getattr(client, "_client", None)
        if not isinstance(inner, httpx.AsyncClient):
            return
        seen: set[int] = set()
        for transport in [inner._transport, *inner._mounts.values()]:
            pool = getattr(transport, "_pool", None)
            for connection in list(getattr(pool, "connections", ())):
                # anyio's normal aclose schedules callbacks on its owner loop.
                # After that loop has closed there is no concurrent I/O, so
                # release the selector transport's socket directly instead.
                _close_orphan_stream(connection, seen)
        inner._state = ClientState.CLOSED
    except Exception:
        logger.debug("Orphaned auxiliary transport cleanup failed", exc_info=True)


async def close_async_clients_for_loop() -> None:
    """Close this disposable loop's cached clients before closing the loop."""
    from agent import auxiliary_client as aux

    loop = asyncio.get_running_loop()
    with aux._client_cache_lock:
        owned = []
        for key, entry in list(aux._client_cache.items()):
            if entry[2] is loop:
                owned.append(entry[0])
                del aux._client_cache[key]
    for client in owned:
        try:
            close = getattr(client, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await asyncio.wait_for(result, timeout=5.0)
        except Exception:
            logger.debug("Auxiliary owner-loop cleanup failed", exc_info=True)
