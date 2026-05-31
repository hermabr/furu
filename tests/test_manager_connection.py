from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from furu.execution.connection import DirectManagerConnection, CloudflareQuickTunnel


def test_direct_manager_connection_returns_local_url() -> None:
    with DirectManagerConnection().connect(local_url="http://127.0.0.1:1234") as url:
        assert url == "http://127.0.0.1:1234"


def test_cloudflare_quick_tunnel_command_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        with CloudflareQuickTunnel(command=()).connect(local_url="http://127.0.0.1:1"):
            pass


def test_cloudflare_quick_tunnel_missing_command_gives_clear_error() -> None:
    with pytest.raises(RuntimeError, match="could not find"):
        with CloudflareQuickTunnel(
            command=("definitely-not-cloudflared-furu-test",)
        ).connect(local_url="http://127.0.0.1:1"):
            pass


def test_cloudflare_quick_tunnel_parses_url_from_output(tmp_path: Path) -> None:
    fake_script = tmp_path / "fake_cloudflared.py"
    argv_file = tmp_path / "argv.json"
    terminated_file = tmp_path / "terminated"
    fake_script.write_text(
        textwrap.dedent(
            f"""
            import json
            import signal
            import sys
            import time
            from pathlib import Path

            argv_file = Path({str(argv_file)!r})
            terminated_file = Path({str(terminated_file)!r})

            def handle_sigterm(signum, frame):
                terminated_file.write_text("terminated", encoding="utf-8")
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, handle_sigterm)
            argv_file.write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
            print("starting fake cloudflared", flush=True)
            print("https://example-furu.trycloudflare.com", flush=True)

            while True:
                time.sleep(1)
            """
        ).lstrip(),
        encoding="utf-8",
    )

    tunnel = CloudflareQuickTunnel(
        command=(sys.executable, str(fake_script)),
        startup_timeout=5,
    )

    with tunnel.connect(local_url="http://127.0.0.1:1234") as url:
        assert url == "https://example-furu.trycloudflare.com"

    assert json.loads(argv_file.read_text(encoding="utf-8")) == [
        "tunnel",
        "--url",
        "http://127.0.0.1:1234",
    ]
    assert terminated_file.exists()


def test_cloudflare_quick_tunnel_timeout_stops_process_and_includes_output(
    tmp_path: Path,
) -> None:
    fake_script = tmp_path / "fake_cloudflared_timeout.py"
    terminated_file = tmp_path / "terminated"
    fake_script.write_text(
        textwrap.dedent(
            f"""
            import signal
            import time
            from pathlib import Path

            terminated_file = Path({str(terminated_file)!r})

            def handle_sigterm(signum, frame):
                terminated_file.write_text("terminated", encoding="utf-8")
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, handle_sigterm)
            print("still starting without a URL", flush=True)

            while True:
                time.sleep(1)
            """
        ).lstrip(),
        encoding="utf-8",
    )

    tunnel = CloudflareQuickTunnel(
        command=(sys.executable, str(fake_script)),
        startup_timeout=0.1,
    )

    with pytest.raises(
        TimeoutError,
        match="did not print a trycloudflare URL.*still starting without a URL",
    ):
        with tunnel.connect(local_url="http://127.0.0.1:1234"):
            pass

    assert terminated_file.exists()


def test_cloudflare_quick_tunnel_early_exit_includes_captured_output(
    tmp_path: Path,
) -> None:
    fake_script = tmp_path / "fake_cloudflared_exit.py"
    fake_script.write_text(
        textwrap.dedent(
            """
            import sys

            print("config error", flush=True)
            raise SystemExit(2)
            """
        ).lstrip(),
        encoding="utf-8",
    )

    tunnel = CloudflareQuickTunnel(
        command=(sys.executable, str(fake_script)),
        startup_timeout=5,
    )

    with pytest.raises(RuntimeError, match="config error"):
        with tunnel.connect(local_url="http://127.0.0.1:1234"):
            pass
