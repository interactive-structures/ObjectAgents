"""
- what are the dependencies we need?
    - opencv?
- how should this piece of code interact with the main task?
    - as an async task? or a thread?
"""

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np



@dataclass
class LatestFrame:
    seq: int = -1
    ts: float = 0.0
    fps: float = 0.0
    frame_rgb: np.ndarray | None = None  # forward ref for type checkers


def frame_to_openai_format(frame_rgb: np.ndarray) -> str:
    """Convert OpenCV frame to base64 encoded format for OpenAI Vision API."""
    # Encode frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame_rgb)

    # Convert to base64 string
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return f"data:image/jpeg;base64,{base64_image}"


class CameraCapture:
    def __init__(
        self, index: int | str, width: int = 640, height: int = 480, sleep: float = 0.1
    ) -> Self:
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.latest_frame = LatestFrame()
        self._ready_event = asyncio.Event()

        # Detect if this is a video file vs camera and adjust timing accordingly
        self.is_video_file = isinstance(index, str)
        if self.is_video_file:

            self.sleep = 1.0 / 20.0  # Default to 20 FPS if unable to detect
               
        else:
            # For camera streams, use provided sleep interval
            self.sleep = sleep
        # # Try AVFoundation first on macOS, then fall back to any backend.
        # self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # self.latest_frame = LatestFrame()
        # self._ready_event = asyncio.Event()

        # self.sleep = sleep

        # if not self.cap.isOpened():
        #     # Retry with CAP_ANY in case AVFoundation-by-index is not supported
        #     self.cap.release()
        #     self.cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            msg = f"Unable to open camera {index}"
            raise RuntimeError(msg)

    async def loop(self):
        while True:
            ok, frame = await asyncio.to_thread(self.cap.read)
            if ok and frame is not None:
                self.latest_frame.frame_rgb = frame
                self.latest_frame.seq += 1
                ts = time.time()
                self.latest_frame.ts = ts

                # Signal that the first frame is ready
                if not self._ready_event.is_set():
                    self._ready_event.set()

            await asyncio.sleep(self.sleep)

    async def ready(self) -> None:
        """Wait until the first frame is captured and processed."""
        await self._ready_event.wait()

    def get_latest_frame(self) -> LatestFrame:
        if self.latest_frame is None:
            raise Exception("is camera loop() function runnning?")
        return self.latest_frame

    def get_latest_frame_rgb(self) -> np.ndarray:
        if self.latest_frame.frame_rgb is None:
            raise Exception("is camera loop() function runnning?")
        return self.latest_frame.frame_rgb

    def get_latest_frame_as_openai_format(self) -> str:
        """Get the latest captured frame."""
        if self.latest_frame.frame_rgb is None:
            raise Exception("is camera loop() function runnning?")
        return frame_to_openai_format(self.latest_frame.frame_rgb)

    def release(self):
        self.cap.release()

# ?
# def _parse_env_indices() -> list[int]:
#     # Supports either CAMERA_INDICES="0,2" or the two specific vars
#     indices: list[int] = []
#     multi = os.getenv("CAMERA_INDICES")
#     if multi:
#         for part in multi.split(","):
#             part = part.strip()
#             if part:
#                 try:
#                     indices.append(int(part))
#                 except ValueError:
#                     pass
#     def _get(var: str) -> int | None:
#         v = os.getenv(var)
#         if v is None or v == "":
#             return None
#         try:
#             return int(v)
#         except ValueError:
#             return None
#     t = _get("TOP_DOWN_CAMERA_INDEX")
#     e = _get("ENV_CAMERA_INDEX")
#     if t is not None:
#         indices.append(t)
#     if e is not None and e != t:
#         indices.append(e)
#     # Keep order and uniqueness
#     seen = set()
#     ordered_unique = []
#     for i in indices:
#         if i not in seen:
#             seen.add(i)
#             ordered_unique.append(i)
#     return ordered_unique


def init_camera(TOP_DOWN_CAMERA_INDEX, ENV_CAMERA_INDEX, agent_loop_debugging, env_video_path) -> tuple[CameraCapture, CameraCapture]:
    # Use env overrides if provided; otherwise the module defaults
    # env_indices = _parse_env_indices()
    # indices: list[int] = env_indices[:2] if len(env_indices) >= 2 else [TOP_DOWN_CAMERA_INDEX, ENV_CAMERA_INDEX]
    
    # top_down_capture = CameraCapture(indices[0])
    # env_capture = CameraCapture(indices[1])
    if agent_loop_debugging:
        top_down_capture = CameraCapture(env_video_path)
        env_capture = CameraCapture(env_video_path)
    else:
        top_down_capture = CameraCapture(TOP_DOWN_CAMERA_INDEX)
        env_capture = CameraCapture(ENV_CAMERA_INDEX)

    return top_down_capture, env_capture


async def main():
    _, env_capture = init_camera(0, 0, True, "test_videos/p1_orgdoc.mp4")

    async def grab_camera_frame():
        # Wait for the camera to be ready before trying to get frames
        print("Waiting for camera to be ready...")
        await env_capture.ready()
        print("Camera is ready! Getting frame...")
        print("latest frame", env_capture.get_latest_frame_as_openai_format())

    tasks = [
        # these two tasks open and run capture on cameras
        env_capture.loop(),
        grab_camera_frame(),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
