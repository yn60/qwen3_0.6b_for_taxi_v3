"""Wrapper around Gymnasium's Taxi-v3 environment with helper utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym

from . import state_utils


_LANDMARK_CODES = ("R", "G", "Y", "B")
_LANDMARK_NAMES = ("Red", "Green", "Yellow", "Blue")


class TaxiEnvironment:
    """Small convenience layer around the classic Taxi-v3 environment."""

    def __init__(self, env_name: str = "Taxi-v3") -> None:
        self._env = gym.make(env_name, render_mode="ansi")
        self._observation: Optional[int] = None
        self._last_render: str = ""

        self._rows: int = 0
        self._cols: int = 0
        self._vertical_walls: list[list[bool]] = []
        self._landmarks: list[Dict[str, Any]] = []
        self._landmark_lookup: Dict[tuple[int, int], Dict[str, Any]] = {}

        self._setup_static_layout()
        self.reset()

    @property
    def observation(self) -> int:
        if self._observation is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._observation

    def reset(self) -> int:
        obs, _ = self._env.reset()
        self._observation = int(obs)
        self._last_render = self._ensure_text(self._env.render())
        return self._observation

    def step(self, action: int):
        obs, reward, terminated, truncated, _ = self._env.step(int(action))
        self._observation = int(obs)
        self._last_render = self._ensure_text(self._env.render())
        done = bool(terminated or truncated)
        return self._observation, float(reward), done

    def sample_action(self) -> int:
        return int(self._env.action_space.sample())

    def decode_state(self, state: Optional[int] = None):
        return state_utils.decode_state(self._state_or_default(state))

    def describe_state(self, state: Optional[int] = None) -> str:
        decoded = self.decode_state(state)
        return state_utils.describe_state_for_llm(decoded)

    def render(self) -> str:
        return self._ensure_text(self._env.render())

    def grid_state(self, decoded_state: Dict[str, int]) -> Dict[str, Any]:
        """Return a rich grid representation for the frontend dashboard."""

        taxi_row = decoded_state["taxi_row"]
        taxi_col = decoded_state["taxi_col"]
        passenger_location = decoded_state["passenger_location"]
        destination_index = decoded_state["destination_index"]

        passenger_in_taxi = passenger_location == 4
        passenger_landmark = None if passenger_in_taxi else self._landmarks[passenger_location]
        destination_landmark = self._landmarks[destination_index]

        cells: list[list[Dict[str, Any]]] = []
        for row in range(self._rows):
            row_cells: list[Dict[str, Any]] = []
            for col in range(self._cols):
                landmark = self._landmark_lookup.get((row, col))
                landmark_payload = None
                if landmark:
                    landmark_payload = {
                        "code": landmark["code"],
                        "name": landmark["name"],
                    }

                cell_payload = {
                    "row": row,
                    "col": col,
                    "walls": self._walls_for_cell(row, col),
                    "landmark": landmark_payload,
                    "is_taxi": row == taxi_row and col == taxi_col,
                    "has_passenger": bool(
                        passenger_landmark
                        and passenger_landmark["row"] == row
                        and passenger_landmark["col"] == col
                    ),
                    "is_destination": destination_landmark["row"] == row
                    and destination_landmark["col"] == col,
                }
                row_cells.append(cell_payload)
            cells.append(row_cells)

        return {
            "rows": self._rows,
            "cols": self._cols,
            "cells": cells,
            "passenger_in_taxi": passenger_in_taxi,
            "passenger_code": passenger_landmark["code"] if passenger_landmark else None,
            "destination_code": destination_landmark["code"],
        }

    def close(self) -> None:
        self._env.close()

    def _state_or_default(self, state: Optional[int]) -> int:
        return int(self.observation if state is None else state)

    def _ensure_text(self, frame) -> str:
        if frame is None:
            return self._last_render
        if isinstance(frame, str):
            self._last_render = frame
        elif hasattr(frame, "__str__"):
            self._last_render = str(frame)
        return self._last_render

    def _setup_static_layout(self) -> None:
        base_env = self._env.unwrapped
        desc = base_env.desc

        self._rows = int(desc.shape[0] - 2)
        self._cols = int((desc.shape[1] - 1) // 2)
        self._vertical_walls = self._compute_vertical_walls(desc)
        self._landmarks = self._compute_landmarks(base_env.locs)
        self._landmark_lookup = {
            (landmark["row"], landmark["col"]): landmark for landmark in self._landmarks
        }

    def _compute_vertical_walls(self, desc) -> list[list[bool]]:
        walls: list[list[bool]] = [[False] * (self._cols - 1) for _ in range(self._rows)]
        for row in range(self._rows):
            ascii_row = desc[1 + row]
            for col in range(self._cols - 1):
                char = ascii_row[2 * col + 2]
                walls[row][col] = char == b"|"
        return walls

    def _compute_landmarks(self, locs) -> list[Dict[str, Any]]:
        landmarks: list[Dict[str, Any]] = []
        for idx, (row, col) in enumerate(locs):
            info = {
                "code": _LANDMARK_CODES[idx],
                "name": _LANDMARK_NAMES[idx],
                "row": int(row),
                "col": int(col),
            }
            landmarks.append(info)
        return landmarks

    def _walls_for_cell(self, row: int, col: int) -> Dict[str, bool]:
        west_wall = col == 0 or (col > 0 and self._vertical_walls[row][col - 1])
        east_wall = col == self._cols - 1 or (
            col < self._cols - 1 and self._vertical_walls[row][col]
        )
        north_wall = row == 0
        south_wall = row == self._rows - 1
        return {
            "north": north_wall,
            "south": south_wall,
            "west": west_wall,
            "east": east_wall,
        }