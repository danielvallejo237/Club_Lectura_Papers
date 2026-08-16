"""Local keyboard play — same CartPole physics as training."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pygame

from cartpole.env import cartpole_physics_from_config, make_env


def _load_config(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_human(env, action: int | None) -> tuple:
    """action 0/1 = push; None = zero force (coast)."""
    base = env.unwrapped
    if action is None:
        old = base.force_mag
        base.force_mag = 0.0
        try:
            return env.step(0)
        finally:
            base.force_mag = old
    return env.step(action)


def play(config_path: str = "configs/cartpole_paperlike.yaml", fps: int = 50, zoom: int = 1) -> None:
    config = _load_config(config_path)
    physics = cartpole_physics_from_config(config)
    x_limit = float(physics["x_limit"])
    angle_limit_deg = float(physics["angle_limit_deg"])

    env = make_env(config, render_mode="rgb_array", disable_time_limit=True)
    tau = float(env.unwrapped.tau)
    clock_fps = fps or int(env.metadata.get("render_fps", 50))

    pygame.init()
    pygame.display.set_caption("CartPole")
    pygame.key.set_repeat(1, max(1, 1000 // clock_fps))
    font = pygame.font.SysFont("dejavusans", 18)

    obs, _ = env.reset()
    score = 0.0
    best_score = 0.0
    done = False
    left_held = False
    right_held = False
    screen: pygame.Surface | None = None
    clock = pygame.time.Clock()

    print(f"CartPole |x|<{x_limit}, fail ±{angle_limit_deg:.0f}°. Hold a/d; release = coast.")
    print("Space/r restart after fall. q quit.")

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
                    left_held = False
                    right_held = False
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    left_held = True
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    right_held = True
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_a, pygame.K_LEFT):
                    left_held = False
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    right_held = False

        if not running:
            break

        pressed = pygame.key.get_pressed()
        left = left_held or bool(pressed[pygame.K_a]) or bool(pressed[pygame.K_LEFT])
        right = right_held or bool(pressed[pygame.K_d]) or bool(pressed[pygame.K_RIGHT])

        if left and not right:
            action: int | None = 0
        elif right and not left:
            action = 1
        else:
            action = None

        if not done:
            obs, reward, terminated, truncated, _ = step_human(env, action)
            score += float(reward)
            done = bool(terminated or truncated)
            if done:
                best_score = max(best_score, score)
                left_held = False
                right_held = False

        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.asarray(frame).swapaxes(0, 1))
            if zoom != 1:
                surf = pygame.transform.scale(surf, (surf.get_width() * zoom, surf.get_height() * zoom))
            if screen is None:
                screen = pygame.display.set_mode((surf.get_width(), surf.get_height() + 70))
            screen.fill((245, 245, 245))
            seconds = score * tau
            if done:
                hint = f"Fell. Space/r = new game. best={best_score:.0f}"
            elif action is None:
                hint = "Coasting — hold a/Left or d/Right to push."
            else:
                hint = "Pushing LEFT" if action == 0 else "Pushing RIGHT"
            angle = float(np.degrees(obs[2]))
            hud = [
                f"score {score:.0f}  (~{seconds:.1f}s)   best {best_score:.0f}   "
                f"x={obs[0]:+.2f}/{x_limit}   angle={angle:+.1f}°",
                hint,
                "HOLD a/Left or d/Right   release = coast   Space/r restart   q quit",
            ]
            y = 8
            for line in hud:
                screen.blit(font.render(line, True, (20, 20, 20)), (10, y))
                y += 22
            screen.blit(surf, (0, 70))
            pygame.display.flip()

        clock.tick(clock_fps)

    pygame.quit()
    env.close()
    print(f"session best score: {best_score:.0f} (~{best_score * tau:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play CartPole locally (display required).")
    parser.add_argument("--config", default="configs/cartpole_paperlike.yaml")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--zoom", type=int, default=1)
    args = parser.parse_args()
    try:
        play(args.config, fps=args.fps, zoom=args.zoom)
    except pygame.error as exc:
        print(f"pygame display error: {exc}")
        print("Needs a local display (WSLg on Windows 11).")
        sys.exit(1)


if __name__ == "__main__":
    main()
