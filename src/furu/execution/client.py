from collections.abc import Iterator
from contextlib import contextmanager

from websockets.headers import build_authorization_basic
from websockets.sync.client import ClientConnection, connect

from furu.resources import ResourceRequest
from furu.worker import protocol


@contextmanager
def worker_client(
    *,
    server_url: str,
    auth_token: str,
    worker: str,
    backend: str,
    resources: ResourceRequest,
) -> Iterator[ClientConnection]:
    with connect(
        server_url,
        additional_headers={
            "Authorization": build_authorization_basic("furu", auth_token)
        },
        max_size=None,
    ) as connection:
        assert isinstance(connection, ClientConnection)
        hello = protocol.HelloMessage(
            version=protocol.PROTOCOL_VERSION,
            worker=worker,
            backend=backend,
            resources=resources,
        )
        connection.send(hello.model_dump_json())
        match protocol.server_message_adapter.validate_json(
            connection.recv(timeout=10)
        ):
            case protocol.WelcomeMessage():
                yield connection
            case protocol.StopMessage(reason=reason):
                raise RuntimeError(reason)
            case unexpected:
                raise RuntimeError(f"expected welcome, got {unexpected.kind!r}")
