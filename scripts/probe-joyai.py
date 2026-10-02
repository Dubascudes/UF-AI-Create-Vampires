"""Send a prerecorded clip through the live JoyAI WebSocket and record generated frames.

Run locally through the SSH tunnel, never on a HiPerGator login node.
"""

import argparse
import asyncio
from contextlib import suppress
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


def extract_frames(video: Path, directory: Path, width: int, height: int, fps: int, limit: int) -> list[Path]:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},scale={width}:{height}",
         "-frames:v", str(limit), "-q:v", "5", str(directory / "frame-%04d.jpg")],
        check=True,
    )
    return sorted(directory.glob("frame-*.jpg"))


async def probe(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        frames = extract_frames(args.video, Path(temporary), args.width, args.height, args.fps, args.frames)
        if not frames:
            raise RuntimeError("No input frames were decoded")
        started = asyncio.Event()
        send_done = asyncio.Event()
        latencies: list[float] = []
        output_times: list[float] = []
        first_sent: float | None = None
        output_meta: dict | None = None
        received = 0
        passthrough = 0
        sent = 0

        async with connect(args.url, max_size=None, ping_timeout=30) as socket:
            async def sender() -> None:
                nonlocal first_sent, sent
                await started.wait()
                interval = 1 / args.fps
                next_at = time.perf_counter()
                last_sent_at = 0.0
                while not send_done.is_set():
                    if sent - received >= 64 and time.perf_counter() - last_sent_at < 1.0:
                        await asyncio.sleep(0.1)
                        next_at = time.perf_counter()
                        continue
                    await asyncio.sleep(max(0, next_at - time.perf_counter()))
                    captured_at = int(time.time() * 1000)
                    if first_sent is None:
                        first_sent = time.perf_counter()
                    sent += 1
                    frame = frames[(sent - 1) % len(frames)]
                    await socket.send(json.dumps({"type": "frame_meta", "seq": sent, "t_capture_ms": captured_at}))
                    await socket.send(frame.read_bytes())
                    last_sent_at = time.perf_counter()
                    next_at += interval

            send_task = asyncio.create_task(sender())
            deadline = time.monotonic() + args.timeout
            try:
                while time.monotonic() < deadline:
                    try:
                        message = await asyncio.wait_for(socket.recv(), timeout=2)
                    except asyncio.TimeoutError:
                        if send_task.done() and send_task.exception():
                            raise send_task.exception()
                        continue
                    if isinstance(message, str):
                        payload = json.loads(message)
                        kind = payload.get("type")
                        if kind == "session_granted":
                            await socket.send(json.dumps({
                                "type": "start", "prompt": args.prompt, "width": args.width, "height": args.height,
                                "fps": args.fps, "num_inference_steps": 2, "use_pe": False,
                                "gate_enabled": False, "freeze_kv_on_static": False,
                                "kv_reset_frames": 0,
                                "input_codec": "mjpeg", "output_codec": "mjpeg", "seed": 42,
                            }))
                        elif kind == "started":
                            started.set()
                        elif kind == "output_frame":
                            output_meta = payload
                        elif kind == "chunk_done":
                            await socket.send(json.dumps({"type": "ack", "recv": received}))
                        elif kind == "error":
                            raise RuntimeError(f"JoyAI server error: {payload.get('message')}")
                        elif kind == "session_timeout":
                            raise RuntimeError(f"JoyAI session timed out: {payload.get('message')}")
                    else:
                        received += 1
                        if output_meta and (output_meta.get("profile") or {}).get("passthrough"):
                            passthrough += 1
                        (args.output / f"frame-{received:04d}.jpg").write_bytes(message)
                        now = time.monotonic()
                        output_times.append(now)
                        if output_meta and output_meta.get("t_capture_ms"):
                            latencies.append(time.time() * 1000 - float(output_meta["t_capture_ms"]))
                        output_meta = None
                        if received >= args.frames:
                            break
            finally:
                send_done.set()
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                with suppress(ConnectionClosed):
                    await socket.send(json.dumps({"type": "stop"}))

        report = {
            "target": f"{args.width}x{args.height}@{args.fps}",
            "input_frames_sent": sent,
            "generated_frames_received": received,
            "declared_passthrough_frames": passthrough,
            "first_frame_delay_s": round(output_times[0] - first_sent, 3) if output_times and first_sent else None,
            "steady_output_fps": round((len(output_times) - 1) / (output_times[-1] - output_times[0]), 2)
            if len(output_times) > 1 and output_times[-1] > output_times[0] else None,
            "last_32_output_fps": round(31 / (output_times[-1] - output_times[-32]), 2)
            if len(output_times) >= 32 and output_times[-1] > output_times[-32] else None,
            "median_capture_to_receive_ms": round(statistics.median(latencies), 1) if latencies else None,
            "last_32_median_capture_to_receive_ms": round(statistics.median(latencies[-32:]), 1)
            if len(latencies) >= 32 else None,
            "max_capture_to_receive_ms": round(max(latencies), 1) if latencies else None,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    parser.add_argument("--output", type=Path, default=Path("runs/joyai-probe"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--prompt", default="Remove every person from the video. Keep the room, lighting, furniture, and objects in place. Photorealistic stable camera.")
    asyncio.run(probe(parser.parse_args()))
