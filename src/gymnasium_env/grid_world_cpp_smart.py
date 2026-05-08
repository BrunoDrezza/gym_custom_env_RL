from typing import Optional
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit as many free cells as possible while avoiding obstacles.
# The reward function is designed to encourage exploration of new cells and
# discourage revisiting already-visited cells.
#
# Reward function (inspired by deep RL approaches to patrolling/coverage problems):
#   - +1.0 for visiting a new (unvisited) cell
#   - -0.3 for revisiting an already-visited cell
#   - -0.1 step penalty to encourage efficiency
#   - +10.0 bonus for achieving full coverage (all free cells visited)
#   - -5.0 penalty when max steps reached without full coverage
#
# The observation space includes:
#   - Agent's (x, y) location (normalized)
#   - Coverage ratio (proportion of free cells visited)
#   - A 3x3 matrix of neighboring cells centered on the agent,
#     where (1,1) is the agent's position and each cell is:
#       0 = free (not yet visited), 1 = obstacle or wall (including out-of-bounds),
#       2 = already visited position.
#     Cells outside the grid boundaries are treated as walls (1).
#
# The episode ends when all free cells are visited or max steps is reached.
#


class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        size: int = 5,
        obs_quantity: int = 3,
        max_steps: int = 200,
    ):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps

        # Track visited cells
        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((5, 5), dtype=int)  # Agora é uma matriz 5x5

        # Observation: Dict com radar 5x5 + Bússola Vetorial
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(  # Sensor de direção para a sujeira mais próxima
                    low=np.array([-1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "neighbors": gym.spaces.Box(
                    low=np.zeros((5, 5), dtype=np.float32),
                    high=np.full((5, 5), 2.0, dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return (
            len(self.visited) / self.total_free_cells
            if self.total_free_cells > 0
            else 1.0
        )

    def _get_obs(self):
        direction, _ = self._get_nearest_unvisited_info()
        return {
            "agent": np.array(
                [
                    self._agent_location[0] / self.size,
                    self._agent_location[1] / self.size,
                    self.coverage_ratio,
                ],
                dtype=np.float32,
            ),
            "target": direction,  # Injetando a bússola no cérebro da rede
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def set_neighbors(self, obstacles_locations):
        # Create a 5x5 matrix centered on the agent's location.
        matrix = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                # j - 2 e i - 2 garantem que a posição [2,2] da matriz seja o agente
                nx = self._agent_location[0] + (j - 2)
                ny = self._agent_location[1] + (i - 2)
                neighbor = np.array([nx, ny])

                # As regras de preenchimento continuam idênticas
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif any(np.array_equal(neighbor, loc) for loc in obstacles_locations):
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
        self._neighbors = matrix

    def _get_nearest_unvisited_info(self):
        # Encontra células livres não visitadas
        all_cells = set((x, y) for x in range(self.size) for y in range(self.size))
        free_cells = all_cells - set(tuple(loc) for loc in self.obstacles_locations)
        unvisited_cells = list(free_cells - self.visited)

        if not unvisited_cells:
            return np.array([0.0, 0.0], dtype=np.float32), 0

        # Distância de Manhattan
        distances = [
            abs(c[0] - self._agent_location[0]) + abs(c[1] - self._agent_location[1])
            for c in unvisited_cells
        ]

        min_idx = np.argmin(distances)
        nearest = unvisited_cells[min_idx]
        min_dist = distances[min_idx]

        # Vetor de direção normalizado pela dimensão do grid
        direction = np.array(
            [
                (nearest[0] - self._agent_location[0]) / self.size,
                (nearest[1] - self._agent_location[1]) / self.size,
            ],
            dtype=np.float32,
        )

        return direction, min_dist

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()

        # Place agent randomly
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles
        for _ in range(self.obs_quantity):
            obstacle_location = self._agent_location.copy()
            while np.array_equal(obstacle_location, self._agent_location) or any(
                np.array_equal(obstacle_location, loc)
                for loc in self.obstacles_locations
            ):
                obstacle_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
            self.obstacles_locations.append(obstacle_location)

        # Mark starting position as visited
        self.visited.add(tuple(self._agent_location))

        self.set_neighbors(self.obstacles_locations)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Pega distância antes do movimento
        _, dist_before = self._get_nearest_unvisited_info()

        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # ... (código de movimento original) ...
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        if any(
            np.array_equal(self._agent_location, loc)
            for loc in self.obstacles_locations
        ):
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        # Pega distância depois do movimento
        _, dist_after = self._get_nearest_unvisited_info()

        # Recompensa base
        reward = -0.1
        # ... (recompensas de is_new_cell, stayed_in_place, etc.) ...

        # --- Injeção de Reward Shaping ---
        # Se reduziu a distância, ganha bônus. Se aumentou, penalidade.
        shaping = (dist_before - dist_after) * 0.1
        reward += shaping

        # ... (resto do método step: terminated, truncated, etc.) ...

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True,
            (0, 0, 0),
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
