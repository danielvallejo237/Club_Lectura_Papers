"""Shared helpers for 3x3 grid GIFs (labels, downscale, streaming writer)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import imageio.v2 as imageio
import numpy as np

# Each grid cell is downscaled to this width (same physics/render config; smaller file + RAM).
GRID_CELL_WIDTH = 360
GRID_FRAME_STRIDE = 2
GRID_FPS = 15
LABEL_FONT_SIZE = 20
LABEL_LINE_HEIGHT = 22
LABEL_PAD = 6


def shrink_frame(frame: np.ndarray, max_width: int = GRID_CELL_WIDTH) -> np.ndarray:
    """Downscale RGB uint8 frame to max_width (keeps aspect ratio)."""
    frame = np.asarray(frame, dtype=np.uint8)
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = max(1, int(round(h * (max_width / w))))
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(frame).resize((max_width, new_h), Image.BILINEAR))
    except ImportError:
        y_idx = np.linspace(0, h - 1, new_h).astype(np.int32)
        x_idx = np.linspace(0, w - 1, max_width).astype(np.int32)
        return frame[y_idx][:, x_idx]


def episode_label_lines(seed: int, score: float, steps: int, end_reason: str) -> list[str]:
    """Two-line overlay: seed/score on line 1, steps/end on line 2."""
    return [
        f"seed={seed}  score={score:.0f}",
        f"steps={steps}  end={end_reason}",
    ]


def label_frame(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Draw a multi-line banner on top of a frame."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=LABEL_FONT_SIZE)
    except TypeError:
        font = ImageFont.load_default()

    banner_h = LABEL_PAD * 2 + LABEL_LINE_HEIGHT * len(lines)
    draw.rectangle((0, 0, img.width, banner_h), fill=(0, 0, 0))
    y = LABEL_PAD
    for line in lines:
        draw.text((LABEL_PAD, y), line, fill=(255, 255, 255), font=font)
        y += LABEL_LINE_HEIGHT
    return np.asarray(img, dtype=np.uint8)


def prepare_labeled_episodes(
    episodes: list[tuple[list[np.ndarray], int, float, int, str]],
    label_lines_fn: Callable[[int, float, int, str], list[str]] = episode_label_lines,
) -> list[list[np.ndarray]]:
    """Label each episode once. Input: (frames, seed, score, steps, end_reason)."""
    labeled: list[list[np.ndarray]] = []
    for frames, seed, score, steps, end_reason in episodes:
        if not frames:
            continue
        lines = label_lines_fn(seed, score, steps, end_reason)
        labeled.append([label_frame(f, lines) for f in frames])
    return labeled


def stream_grid_gif(
    labeled: list[list[np.ndarray]],
    grid_size: int,
    path: Path,
    fps: int = GRID_FPS,
) -> Path:
    """Tile episodes into a grid GIF, writing one composite frame at a time."""
    if len(labeled) != grid_size * grid_size:
        raise ValueError(f"expected {grid_size * grid_size} episodes, got {len(labeled)}")
    max_len = max(len(frames) for frames in labeled)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gif_path = path.with_suffix(".gif")

    with imageio.get_writer(gif_path, mode="I", fps=fps, loop=0) as writer:
        for t in range(max_len):
            rows: list[np.ndarray] = []
            for row in range(grid_size):
                cells: list[np.ndarray] = []
                for col in range(grid_size):
                    frames = labeled[row * grid_size + col]
                    cells.append(frames[min(t, len(frames) - 1)])
                rows.append(np.concatenate(cells, axis=1))
            writer.append_data(np.concatenate(rows, axis=0))
    return gif_path
