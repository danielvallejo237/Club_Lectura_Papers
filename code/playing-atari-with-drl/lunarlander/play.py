"""Local keyboard play for LunarLander-v3 (discrete actions)."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pygame

from lunarlander.env import make_env


def _load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _action_from_keys(left: bool, main: bool, right: bool) -> int:
    """Map held keys to discrete LunarLander action (0=noop, 1=left, 2=main, 3=right)."""
    if main and not left and not right:
        return 2
    if left and not right:
        return 1
    if right and not left:
        return 3
    if main and left and not right:
        return 1
    if main and right and not left:
        return 3
    return 0


def play(
    config_path: str = "configs/lunarlander_baseline.yaml",
    fps: int = 50,
    zoom: int = 2,
) -> None:
    config = _load_config(config_path)
    config.setdefault("env_id", "LunarLander-v3")

    env = make_env(config, render_mode="rgb_array", disable_time_limit=False)
    clock_fps = fps or int(env.metadata.get("render_fps", 50))

    pygame.init()
    pygame.display.set_caption("LunarLander")
    pygame.key.set_repeat(1, max(1, 1000 // clock_fps))
    font = pygame.font.SysFont("dejavusans", 18)

    obs, _ = env.reset()
    score = 0.0
    best_score = float("-inf")
    done = False
    screen: pygame.Surface | None = None
    clock = pygame.time.Clock()

    print("LunarLander — land on the pad with low speed and upright orientation.")
    print("Hold a/Left = left engine, w/Up = main, d/Right = right. Release = coast.")
    print("Space/r restart after landing. q quit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif done and event.key in (pygame.K_SPACE, pygame.K_r):
                    best_score = max(best_score, score)
                    obs, _ = env.reset()
                    score = 0.0
                    done = False

        if not running:
            break

        pressed = pygame.key.get_pressed()
        left = bool(pressed[pygame.K_a]) or bool(pressed[pygame.K_LEFT])
        main = bool(pressed[pygame.K_w]) or bool(pressed[pygame.K_UP])
        right = bool(pressed[pygame.K_d]) or bool(pressed[pygame.K_RIGHT])
        action = _action_from_keys(left, main, right)

        if not done:
            obs, reward, terminated, truncated, _ = env.step(action)
            score += float(reward)
            done = bool(terminated or truncated)
            if done:
                best_score = max(best_score, score)

        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.asarray(frame).swapaxes(0, 1))
            if zoom != 1:
                surf = pygame.transform.scale(
                    surf, (surf.get_width() * zoom, surf.get_height() * zoom)
                )
            if screen is None:
                screen = pygame.display.set_mode((surf.get_width(), surf.get_height() + 88))
            screen.fill((245, 245, 245))
            if done:
                hint = f"Episode over. Space/r = new game. best={best_score:.0f}"
            elif action == 0:
                hint = "Coasting — hold a/w/d or arrow keys to fire engines."
            elif action == 1:
                hint = "Left engine"
            elif action == 2:
                hint = "Main engine"
            else:
                hint = "Right engine"
            hud = [
                f"score {score:.1f}   best {best_score:.1f}   "
                f"x={obs[0]:+.2f} y={obs[1]:+.2f} vx={obs[2]:+.2f} vy={obs[3]:+.2f}",
                hint,
                "a/Left left   w/Up main   d/Right right   release=coast   Space/r restart   q quit",
            ]
            y = 8
            for line in hud:
                screen.blit(font.render(line, True, (20, 20, 20)), (10, y))
                y += 24
            screen.blit(surf, (0, 88))
            pygame.display.flip()

        clock.tick(clock_fps)

    pygame.quit()
    env.close()
    if best_score > float("-inf"):
        print(f"session best score: {best_score:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play LunarLander locally (display required).")
    parser.add_argument("--config", default="configs/lunarlander_baseline.yaml")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--zoom", type=int, default=2)
    args = parser.parse_args()
    try:
        play(args.config, fps=args.fps, zoom=args.zoom)
    except pygame.error as exc:
        print(f"pygame display error: {exc}")
        print("Needs a local display (WSLg on Windows 11, or X11 forwarding).")
        sys.exit(1)


if __name__ == "__main__":
    main()
