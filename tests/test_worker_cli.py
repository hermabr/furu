from __future__ import annotations

import pytest

import furu.worker.cli as cli_module
from furu.worker import cli as cli


def test_cli_main_invokes_worker_loop_with_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_worker_loop(*, server_url: str, auth_token: str) -> None:
        captured["server_url"] = server_url
        captured["auth_token"] = auth_token

    monkeypatch.setattr(cli_module, "worker_loop", fake_worker_loop)

    rc = cli.main(
        [
            "--server-url",
            "http://manager.test:8080",
            "--auth-token",
            "tok-123",
        ]
    )

    assert rc == 0
    assert captured == {
        "server_url": "http://manager.test:8080",
        "auth_token": "tok-123",
    }


def test_cli_main_requires_auth_token() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--server-url", "http://manager.test:8080"])


def test_cli_main_requires_server_url() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--auth-token", "tok"])
