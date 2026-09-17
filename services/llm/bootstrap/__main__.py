"""Small production entry point; configuration remains the typed schema."""
from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path

from .config import load_config
from .runtime import BootstrapRuntime, RuntimeOptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the offline runtime: [serve] CONFIG")
    parser.add_argument("command_or_config")
    parser.add_argument("config", type=Path, nargs="?")
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--shutdown-grace-seconds", type=float, default=60.0)
    parser.add_argument("--host-pid-namespace", action="store_true",
                        help="attest that /proc and NVML use the host PID namespace")
    args = parser.parse_args()
    # Preserve the original ``bootstrap CONFIG`` form while accepting the
    # explicit production ``bootstrap serve CONFIG`` form.
    if args.config is None:
        command, config = "serve", Path(args.command_or_config)
    else:
        command, config = args.command_or_config, args.config
    if command != "serve":
        parser.error("command must be serve")
    runtime = BootstrapRuntime(load_config(config), RuntimeOptions(
        args.state_dir, args.result_dir, args.host, args.port,
        shutdown_grace_seconds=args.shutdown_grace_seconds,
        host_pid_namespace=args.host_pid_namespace))
    async def serve() -> None:
        loop = asyncio.get_running_loop()
        stopping = asyncio.Event()
        for value in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(value, stopping.set)
        task = asyncio.create_task(runtime.run())
        signal_task = asyncio.create_task(stopping.wait())
        try:
            await asyncio.wait((task, signal_task), return_when=asyncio.FIRST_COMPLETED)
            if stopping.is_set():
                await runtime.stop()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # stop cancels an in-flight startup or daemon monitor
            else:
                await task
                raise RuntimeError("owned Ollama daemon was lost")
        finally:
            signal_task.cancel()
            await asyncio.gather(signal_task, return_exceptions=True)
            for value in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(value)
    asyncio.run(serve())


if __name__ == "__main__":
    main()
