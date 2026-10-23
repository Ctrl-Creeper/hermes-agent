"""Real local TCP tests: no provider calls and no garbage-collection dependency."""
import asyncio
import queue
import socketserver
import threading

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agent import auxiliary_client as aux


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(aux, '_client_cache', {})


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.recv(8192)
        self.request.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}')
        self.server.accepted.put(self.request)
        self.server.release.wait(10)


@pytest.fixture
def endpoint():
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.accepted = queue.Queue()
    server.release = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/', server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_sync_cache_cleanup_closes_real_socket(endpoint):
    url, server = endpoint
    client = OpenAI(api_key='local-test', base_url=url,
                    http_client=httpx.Client(trust_env=False))
    try:
        client._client.get(url)
        peer = server.accepted.get(timeout=2)
        aux._close_cached_client(client)
        peer.settimeout(1)
        assert peer.recv(1) == b'', 'cache cleanup left the TCP connection open'
    finally:
        # Bypass a falsely CLOSED client state so a failed test releases FDs.
        client._client._transport.close()


@pytest.mark.parametrize('proxy_mount', [False, True])
def test_stale_async_cache_cleanup_releases_socket_after_loop_closed(endpoint, proxy_mount):
    url, server = endpoint
    loop = asyncio.new_event_loop()
    client = AsyncOpenAI(api_key='local-test', base_url=url,
                         http_client=httpx.AsyncClient(
                             trust_env=False, proxy=url if proxy_mount else None,
                         ))
    key = ('socket-cleanup-test',)
    try:
        loop.run_until_complete(client._client.get(url))
        peer = server.accepted.get(timeout=2)
        loop.close()
        aux._client_cache[key] = (client, 'local', loop)
        aux.cleanup_stale_async_clients()
        assert key not in aux._client_cache
        peer.settimeout(1)
        assert peer.recv(1) == b'', 'stale cleanup only marked the client CLOSED'
    finally:
        aux._client_cache.pop(key, None)
        if not loop.is_closed():
            loop.run_until_complete(client.close())
            loop.close()


def test_owner_loop_cleanup_closes_before_loop_exit(endpoint):
    url, server = endpoint

    async def scenario():
        client = AsyncOpenAI(api_key='local-test', base_url=url,
                             http_client=httpx.AsyncClient(trust_env=False))
        try:
            await client._client.get(url)
            peer = server.accepted.get(timeout=2)
            key = ('owner-loop-cleanup-test',)
            aux._client_cache[key] = (client, 'local', asyncio.get_running_loop())
            from agent.auxiliary_client_cleanup import close_async_clients_for_loop
            await close_async_clients_for_loop()
            assert key not in aux._client_cache
            peer.settimeout(1)
            assert peer.recv(1) == b''
        finally:
            aux._client_cache.pop(('owner-loop-cleanup-test',), None)
            await client.close()

    asyncio.run(scenario())

def test_disposable_tool_loop_releases_its_auxiliary_connections(endpoint):
    from model_tools import _run_async

    url, server = endpoint

    async def tool():
        client = AsyncOpenAI(api_key='local-test', base_url=url,
                             http_client=httpx.AsyncClient(trust_env=False))
        await client._client.get(url)
        loop = asyncio.get_running_loop()
        aux._client_cache[('bridge-cleanup-test',)] = (client, 'local', loop)
        return client, loop

    async def gateway():
        # Exercises the disposable-worker branch used from a running loop.
        client, loop = _run_async(tool())
        assert loop.is_closed()
        assert client.is_closed()
        assert not aux._client_cache
        peer = server.accepted.get(timeout=2)
        peer.settimeout(1)
        assert peer.recv(1) == b''

    asyncio.run(gateway())


def test_stale_cleanup_does_not_close_a_live_loop_client(endpoint):
    url, server = endpoint

    async def scenario():
        client = AsyncOpenAI(api_key='local-test', base_url=url,
                             http_client=httpx.AsyncClient(trust_env=False))
        try:
            await client._client.get(url)
            aux._client_cache[('live-client-test',)] = (
                client, 'local', asyncio.get_running_loop(),
            )
            aux.cleanup_stale_async_clients()
            assert ('live-client-test',) in aux._client_cache
            assert not client.is_closed()
            peer = server.accepted.get(timeout=2)
            peer.settimeout(0.05)
            with pytest.raises(TimeoutError):
                peer.recv(1)
        finally:
            await client.close()

    asyncio.run(scenario())
