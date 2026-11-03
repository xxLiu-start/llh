"""Temporary bridge-bound TCP relay for Docker builds using a loopback proxy."""

from __future__ import annotations

import contextlib
import os
import select
import socket
import socketserver
import subprocess
import threading
from collections.abc import Iterator
from urllib.parse import urlsplit


class _ProxyRelay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], upstream: tuple[str, int]):
        self.upstream = upstream
        super().__init__(address, _RelayConnection)


class _RelayConnection(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            with socket.create_connection(self.server.upstream, timeout=10) as remote:
                peers = (self.request, remote)
                while True:
                    readable, _, _ = select.select(peers, [], [], 60)
                    if not readable:
                        continue
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        (remote if source is self.request else self.request).sendall(data)
        except (ConnectionError, OSError):
            return


def _docker_bridge_gateway() -> str:
    result = subprocess.run(
        ["docker", "network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}"],
        check=True, capture_output=True, text=True,
    )
    gateway = result.stdout.strip()
    if not gateway:
        raise RuntimeError("Docker bridge has no gateway address for the build proxy")
    return gateway


@contextlib.contextmanager
def docker_build_proxy() -> Iterator[str | None]:
    """Expose the host HTTP proxy to bridge builds, only for this Harbor trial."""
    proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if not proxy_url:
        yield None
        return
    parsed = urlsplit(proxy_url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        raise RuntimeError("Docker build proxy requires an HTTP(S)_PROXY URL with http://host:port")
    # A stable port keeps Docker's build-arg cache key stable across baseline
    # and candidate replays. Training itself is sequential for the shared bank.
    listen_port = int(os.getenv("LH_DOCKER_BUILD_PROXY_PORT", "17897"))
    server = _ProxyRelay((_docker_bridge_gateway(), listen_port), (parsed.hostname, parsed.port))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield f"http://{server.server_address[0]}:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
