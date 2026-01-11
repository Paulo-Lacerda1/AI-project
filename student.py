from __future__ import annotations

import asyncio
import getpass
import json
import os
from typing import Any, Dict, Optional, Tuple

import websockets

from consts import COOL_DOWN

DANGER_ZONE_HORIZONTAL = 5        # Alcance horizontal da centopeia para entrar em perigo.
DANGER_ZONE_VERTICAL = 1          # Alcance vertical da centopeia para entrar em perigo.
FLEE_SAFETY_MARGIN = 2            # Margem vertical mínima acima da pulga para engajar.     #aqui era 4 
FLEE_ESCAPE_BLOCK_DISTANCE = 6    # Evita ficar alinhado muito perto abaixo da pulga.
MAX_OBSTACLES_TO_PURSUE = 3       # Máximo de cogumelos no caminho ao perseguir pulga.
FULL_ESCAPE_BOTTOM_ROWS = 2       # Linhas do fundo que ativam fuga total.                  #AQUI ERA 4
FULL_ESCAPE_UPWARD_BONUS = 500.0  # Bónus por subir em modo de fuga total.
SAFE_DESCENT_DISTANCE = 8         # Distância Manhattan mínima para descer seguro.
SPIDER_ADJACENT_DISTANCE = 1      # Spider adjacente é considerado inseguro.
SPIDER_DANGER_DISTANCE = 3        # Proximidade do spider ativa evasão.
SPIDER_ALERT_DISTANCE = SPIDER_DANGER_DISTANCE + 1  # Evitar o spider com antecedência.


class SmartCalculatorFarmer:
    def __init__(self) -> None:
        self.map_width = 40
        self.map_height = 24
        self.mushrooms = set()
        self.last_shot_frame = -999
        self.current_frame = 0
        self.last_px = -1
        self.tactical_phase = "normal"
        self.pending_up_escape = 0
        self.spider_pos = None

    def update_state(self, state: Dict[str, Any]) -> None:
        if "size" in state:
            self.map_width, self.map_height = state["size"]
        self.mushrooms = {tuple(m["pos"]) for m in state.get("mushrooms", [])}
        self.current_frame = state.get("step", 0)

    def can_shoot(self) -> bool:
        return (self.current_frame - self.last_shot_frame) >= COOL_DOWN

    def _is_position_valid(self, x: int, y: int) -> bool:
        if self.spider_pos is not None and (x, y) == self.spider_pos:
            return False
        return (
            0 <= x < self.map_width
            and 0 <= y < self.map_height
            and (x, y) not in self.mushrooms
        )

    def _is_path_clear_vertical(self, x: int, y_from: int, y_to: int) -> bool:
        if y_from == y_to:
            return True
        start = min(y_from, y_to)
        end = max(y_from, y_to)
        for y in range(start + 1, end):
            if (x, y) in self.mushrooms:
                return False
        return True

    def _snake_in_critical_zone(self, centipedes_positions) -> bool:
        critical_row = self.map_height - FULL_ESCAPE_BOTTOM_ROWS
        for _, cy in centipedes_positions:
            if cy >= critical_row:
                return True
        return False

    def _evaluate_escape_position(
        self, x, y, centipedes_positions, flee_pos=None, player_y=None, spider_pos=None
    ) -> float:
        if not self._is_position_valid(x, y):
            return -1000
        if self._is_flee_escape_blocked(x, y, flee_pos):
            return -1000
        if flee_pos is not None and (x, y) == flee_pos:
            return -1000
        if spider_pos is not None:
            sx, sy = spider_pos
            spider_dist = abs(x - sx) + abs(y - sy)
            if spider_dist <= SPIDER_ADJACENT_DISTANCE:
                return -1000

        if centipedes_positions:
            danger_dist = min(abs(x - cx) + abs(y - cy) for cx, cy in centipedes_positions)
        else:
            danger_dist = 100

        flea_penalty = 0
        if flee_pos is not None:
            fx, fy = flee_pos
            if x == fx and y > fy:
                vertical_dist = y - fy
                if vertical_dist < 6:
                    flea_penalty = (6 - vertical_dist) * 50

        edge_dist = min(x, self.map_width - 1 - x)

        free_spaces = 0
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if self._is_position_valid(x + dx, y + dy):
                free_spaces += 1

        full_escape_bonus = 0.0
        if player_y is not None and self._snake_in_critical_zone(centipedes_positions):
            if y < player_y:
                full_escape_bonus = FULL_ESCAPE_UPWARD_BONUS

        spider_penalty = 0.0
        if spider_pos is not None:
            sx, sy = spider_pos
            spider_dist = abs(x - sx) + abs(y - sy)
            if spider_dist <= SPIDER_DANGER_DISTANCE:
                spider_penalty = (SPIDER_DANGER_DISTANCE + 1 - spider_dist) * 200

        return (
            danger_dist * 10
            + free_spaces * 5
            + edge_dist * 2
            - flea_penalty
            - spider_penalty
            + full_escape_bonus
        )

    def _is_flee_escape_blocked(self, x, y, flee_pos) -> bool:
        if flee_pos is None:
            return False
        fx, fy = flee_pos
        if x != fx:
            return False
        if y < fy:
            return False
        return (y - fy) <= FLEE_ESCAPE_BLOCK_DISTANCE

    def _get_lowest_centipede_row(self, centipedes_positions):
        if not centipedes_positions:
            return None
        return max(cy for _, cy in centipedes_positions)

    def _is_centipede_on_last_line(self, centipedes_positions) -> bool:
        last_row = self.map_height - 1
        return any(cy == last_row for _, cy in centipedes_positions)

    def _is_centipede_on_row22(self, centipedes_positions) -> bool:
        row22 = self.map_height - 2
        return any(cy == row22 for _, cy in centipedes_positions)

    def _is_upward_safe(self, px, py, centipedes_positions) -> bool:
        if py <= 0:
            return False
        for cx, cy in centipedes_positions:
            if cy < py and (py - cy) <= 2 and abs(cx - px) <= 1:
                return False
        return True

    def is_safe_to_engage(self, px, py, fx, fy) -> bool:
        dist_y = py - fy
        if dist_y < FLEE_SAFETY_MARGIN:
            return False

        dist_x = abs(fx - px)
        obstacles = 0
        step = 1 if fx > px else -1
        if dist_x > 0:
            for x in range(px + step, fx + step, step):
                if (x, py) in self.mushrooms:
                    obstacles += 1

        travel_time = dist_x + (obstacles * (COOL_DOWN + 1))
        cooldown_remaining = max(0, (self.last_shot_frame + COOL_DOWN) - self.current_frame)
        time_until_shot = max(travel_time, cooldown_remaining)

        future_flee_y = fy + time_until_shot
        if future_flee_y >= (py - FLEE_SAFETY_MARGIN):
            return False

        if obstacles > MAX_OBSTACLES_TO_PURSUE:
            return False

        return True

    def is_safe_to_descend(self, px, py, centipedes) -> bool:
        if not centipedes:
            return True

        for cent in centipedes:
            for seg in cent.get("body", []):
                cx, cy = seg

                dist_x = abs(px - cx)
                dist_y = abs(py - cy)
                manhattan_dist = dist_x + dist_y

                if manhattan_dist < SAFE_DESCENT_DISTANCE:
                    return False

                if cy > py and manhattan_dist < SAFE_DESCENT_DISTANCE + 2:
                    return False

        return True

    def get_action(self, state: Dict[str, Any]) -> str:
        bug = state.get("bug_blaster")
        if not bug or not bug.get("alive"):
            return ""

        px, py = bug["pos"]

        flee = state.get("flee")
        flee_pos = None
        if flee and flee.get("alive"):
            flee_pos = tuple(flee["pos"])

        spider = state.get("spider")
        spider_pos = None
        if spider and spider.get("alive", True):
            spider_pos = tuple(spider["pos"])
        self.spider_pos = spider_pos

        centipedes = state.get("centipedes", [])
        all_centipede_positions = []
        centipede_heads = [tuple(c["body"][-1]) for c in centipedes if c.get("body")]
        for cent in centipedes:
            all_centipede_positions.extend([tuple(seg) for seg in cent.get("body", [])])

        centipede_positions_set = set(all_centipede_positions)
        spider_dist = None
        if spider_pos is not None:
            spider_dist = abs(px - spider_pos[0]) + abs(py - spider_pos[1])
        spider_close = (
            spider_dist is not None
            and spider_dist <= SPIDER_ALERT_DISTANCE
        )

        if spider_dist is not None and spider_dist <= SPIDER_DANGER_DISTANCE:
            escape_moves = []
            for key, dx, dy in [("a", -1, 0), ("d", 1, 0), ("w", 0, -1), ("s", 0, 1)]:
                nx, ny = px + dx, py + dy
                if not self._is_position_valid(nx, ny):
                    continue
                if (nx, ny) in centipede_positions_set:
                    continue
                score = self._evaluate_escape_position(
                    nx, ny, all_centipede_positions, flee_pos, py, spider_pos
                )
                escape_moves.append((score, key))
            if escape_moves:
                best_score, best_move = max(escape_moves, key=lambda item: item[0])
                if best_score > -1000:
                    return best_move

        if flee_pos is not None:
            fx, fy = flee_pos
            vertical_gap = py - fy
            if fx == px and 0 < vertical_gap <= FLEE_SAFETY_MARGIN:
                if self.can_shoot() and self._is_path_clear_vertical(px, py, fy):
                    self.last_shot_frame = self.current_frame
                    return "A"
                dodge_moves = []
                for key, dx in [("a", -1), ("d", 1)]:
                    nx = px + dx
                    if not self._is_position_valid(nx, py):
                        continue
                    if (nx, py) in centipede_positions_set:
                        continue
                    score = self._evaluate_escape_position(
                        nx, py, all_centipede_positions, flee_pos, py, spider_pos
                    )
                    dodge_moves.append((score, key))
                if dodge_moves:
                    best_score, best_move = max(dodge_moves, key=lambda item: item[0])
                    if best_score > -1000:
                        return best_move
                if (
                    self._is_position_valid(px, py - 1)
                    and (px, py - 1) not in centipede_positions_set
                    and self._is_upward_safe(px, py, all_centipede_positions)
                    and (px, py - 1) != flee_pos
                ):
                    return "w"

        if centipedes:
            any_below = any(cent_y > py for _, cent_y in centipede_heads)
            any_above = any(cent_y < py for _, cent_y in centipede_heads)
            in_danger = False
            for cx, cy in all_centipede_positions:
                dist_x = abs(px - cx)
                dist_y = abs(py - cy)
                if dist_y <= DANGER_ZONE_VERTICAL and dist_x <= DANGER_ZONE_HORIZONTAL:
                    in_danger = True
                    break
            if spider_close:
                in_danger = True

            if in_danger:
                if (
                    self.pending_up_escape > 0
                    and self._is_position_valid(px, py - 1)
                    and (px, py - 1) not in all_centipede_positions
                    and self._is_upward_safe(px, py, all_centipede_positions)
                ):
                    self.pending_up_escape = 0
                    return "w"
                self.pending_up_escape = 0

                in_full_escape_mode = self._snake_in_critical_zone(all_centipede_positions)
                if (
                    in_full_escape_mode
                    and self._is_position_valid(px, py - 1)
                    and self._is_upward_safe(px, py, all_centipede_positions)
                ):
                    score_up = self._evaluate_escape_position(
                        px, py - 1, all_centipede_positions, flee_pos, py, spider_pos
                    )
                    if score_up > 400:
                        return "w"

                escape_options = []
                if self._is_position_valid(px + 1, py):
                    score = self._evaluate_escape_position(
                        px + 1, py, all_centipede_positions, flee_pos, py, spider_pos
                    )
                    escape_options.append(("d", score))
                if self._is_position_valid(px - 1, py):
                    score = self._evaluate_escape_position(
                        px - 1, py, all_centipede_positions, flee_pos, py, spider_pos
                    )
                    escape_options.append(("a", score))

                if escape_options:
                    best_move, best_score = max(escape_options, key=lambda x: x[1])
                    if best_score > 0:
                        return best_move
                    if any_above and best_score > -1000:
                        self.pending_up_escape = 1
                        return best_move

                vertical_options = []
                if not centipede_heads:
                    return ""
                closest_cent = min(centipede_heads, key=lambda c: abs(c[0] - px) + abs(c[1] - py))
                _, cy = closest_cent

                if any_below or (any_above and cy < py):
                    if (
                        self._is_position_valid(px, py - 1)
                        and self._is_upward_safe(px, py, all_centipede_positions)
                    ):
                        score = self._evaluate_escape_position(
                            px, py - 1, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        vertical_options.append(("w", score))

                if not any_below and cy < py:
                    dist_y = abs(py - cy)
                    if dist_y > 2 and self._is_position_valid(px, py + 1):
                        score = self._evaluate_escape_position(
                            px, py + 1, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        vertical_options.append(("s", score))

                if vertical_options:
                    best_move, best_score = max(vertical_options, key=lambda x: x[1])
                    if best_score > 0:
                        return best_move

                if self._is_flee_escape_blocked(px, py, flee_pos):
                    desperation_options = []
                    if self._is_position_valid(px + 1, py) and not self._is_flee_escape_blocked(px + 1, py, flee_pos):
                        score = self._evaluate_escape_position(
                            px + 1, py, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        desperation_options.append(("d", score))
                    if self._is_position_valid(px - 1, py) and not self._is_flee_escape_blocked(px - 1, py, flee_pos):
                        score = self._evaluate_escape_position(
                            px - 1, py, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        desperation_options.append(("a", score))
                    if desperation_options:
                        best_move, _ = max(desperation_options, key=lambda x: x[1])
                        return best_move

                return ""

        is_moving = (px != self.last_px)
        self.last_px = px

        if all_centipede_positions:
            lowest_cent_row = self._get_lowest_centipede_row(all_centipede_positions)
            centipede_in_last_4 = self._snake_in_critical_zone(all_centipede_positions)
            centipede_on_last_line = self._is_centipede_on_last_line(all_centipede_positions)
            centipede_on_row22 = self._is_centipede_on_row22(all_centipede_positions)

            if centipede_in_last_4:
                if self.tactical_phase == "row22_wait" and py == self.map_height - 2:
                    if centipede_on_row22:
                        self.tactical_phase = "row23_wait"
                        last_row = self.map_height - 1
                        if py < last_row and self._is_position_valid(px, py + 1):
                            return "s"

                if (
                    not centipede_on_last_line
                    and lowest_cent_row < self.map_height - 1
                    and self.tactical_phase not in ["row22_wait", "row23_wait"]
                ):
                    self.tactical_phase = "middle_wait"
                    middle_x = self.map_width // 2

                    if flee_pos is not None:
                        fx, fy = flee_pos
                        if fx == px and fy < py:
                            dodge_options = []
                            if self._is_position_valid(px + 1, py):
                                score = self._evaluate_escape_position(
                                    px + 1, py, all_centipede_positions, flee_pos, py, spider_pos
                                )
                                dodge_options.append(("d", score))
                            if self._is_position_valid(px - 1, py):
                                score = self._evaluate_escape_position(
                                    px - 1, py, all_centipede_positions, flee_pos, py, spider_pos
                                )
                                dodge_options.append(("a", score))
                            if dodge_options:
                                best_move, _ = max(dodge_options, key=lambda x: x[1])
                                return best_move
                            if self.can_shoot() and self._is_path_clear_vertical(px, py, fy):
                                self.last_shot_frame = self.current_frame
                                return "A"

                    if px < middle_x:
                        if self._is_position_valid(px + 1, py):
                            return "d"
                    elif px > middle_x:
                        if self._is_position_valid(px - 1, py):
                            return "a"
                    return "s"

                if centipede_on_last_line:
                    self.tactical_phase = "row22_wait"
                    row22 = self.map_height - 2

                    if py > row22:
                        if self._is_position_valid(px, py - 1):
                            return "w"
                        middle_x = self.map_width // 2
                        if px < middle_x and self._is_position_valid(px + 1, py):
                            return "d"
                        if px > middle_x and self._is_position_valid(px - 1, py):
                            return "a"
                        return "s"

                    if py == row22:
                        middle_x = self.map_width // 2
                        if px < middle_x and self._is_position_valid(px + 1, py):
                            return "d"
                        if px > middle_x and self._is_position_valid(px - 1, py):
                            return "a"
                        return "s"

                if self.tactical_phase == "row23_wait":
                    last_row = self.map_height - 1
                    if py < last_row and self._is_position_valid(px, py + 1):
                        return "s"

                    middle_x = self.map_width // 2
                    if px < middle_x and self._is_position_valid(px + 1, py):
                        return "d"
                    if px > middle_x and self._is_position_valid(px - 1, py):
                        return "a"
                    return "s"
            else:
                self.tactical_phase = "normal"

        if self.tactical_phase == "normal":
            if spider_close:
                spider_moves = []
                for key, dx, dy in [("a", -1, 0), ("d", 1, 0), ("w", 0, -1)]:
                    nx, ny = px + dx, py + dy
                    if not self._is_position_valid(nx, ny):
                        continue
                    if (nx, ny) in centipede_positions_set:
                        continue
                    score = self._evaluate_escape_position(
                        nx, ny, all_centipede_positions, flee_pos, py, spider_pos
                    )
                    spider_moves.append((score, key))
                if spider_moves:
                    best_score, best_move = max(spider_moves, key=lambda item: item[0])
                    if best_score > -1000:
                        return best_move
            if py < self.map_height - 1:
                if (px, py + 1) in self.mushrooms:
                    return ""
                if self.is_safe_to_descend(px, py, centipedes):
                    return "s"
                if all_centipede_positions:
                    lateral_options = []
                    if self._is_position_valid(px + 1, py):
                        score = self._evaluate_escape_position(
                            px + 1, py, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        lateral_options.append(("d", score))

                    if self._is_position_valid(px - 1, py):
                        score = self._evaluate_escape_position(
                            px - 1, py, all_centipede_positions, flee_pos, py, spider_pos
                        )
                        lateral_options.append(("a", score))

                    if lateral_options:
                        best_move, best_score = max(lateral_options, key=lambda x: x[1])
                        if best_score > 0:
                            return best_move
                return ""

        target_x = self.map_width // 2
        target_is_flee = False
        in_critical_escape = (
            all_centipede_positions and self._snake_in_critical_zone(all_centipede_positions)
        )

        if flee and flee.get("alive"):
            fx, fy = flee["pos"]
            if spider_close:
                target_x = self.map_width // 2
            elif self.is_safe_to_engage(px, py, fx, fy):
                target_x = fx
                target_is_flee = True
            else:
                target_x = self.map_width // 2

        delta = target_x - px

        if delta == 0:
            in_tactical_phase = self.tactical_phase != "normal"
            if target_is_flee and self.can_shoot() and not in_critical_escape and not in_tactical_phase:
                if is_moving:
                    return "s"
                if flee and flee.get("alive"):
                    fx, fy = flee["pos"]
                    dist_y = py - fy
                    if dist_y % 2 != 0:
                        return "s"
                self.last_shot_frame = self.current_frame
                return "A"
            return "s"

        if delta > 0:
            if (px + 1, py) in self.mushrooms:
                in_tactical_phase = self.tactical_phase != "normal"
                if target_is_flee and self.can_shoot() and not in_critical_escape and not in_tactical_phase:
                    self.last_shot_frame = self.current_frame
                    return "A"
                return "s"
            return "d"

        if delta < 0:
            if (px - 1, py) in self.mushrooms:
                in_tactical_phase = self.tactical_phase != "normal"
                if target_is_flee and self.can_shoot() and not in_critical_escape and not in_tactical_phase:
                    self.last_shot_frame = self.current_frame
                    return "A"
                return "s"
            return "a"

        return ""


class FarmAgent:
    def __init__(self) -> None:
        self.farmer = SmartCalculatorFarmer()

    def process_server_update(self, state: Dict[str, Any]) -> Optional[str]:
        if "size" in state:
            self.farmer.update_state(state)
            return None
        if "highscores" in state:
            return None
        self.farmer.update_state(state)
        return self.farmer.get_action(state)


async def agent_loop(server_address: str = "localhost:8000", agent_name: str = "farm") -> None:
    agent = FarmAgent()
    async with websockets.connect(
        f"ws://{server_address}/player", ping_interval=None
    ) as websocket:
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        try:
            async for raw_message in websocket:
                state = json.loads(raw_message)
                command = agent.process_server_update(state)

                if "highscores" in state:
                    break

                if command is None:
                    continue

                await websocket.send(json.dumps({"cmd": "key", "key": command}))
        except websockets.exceptions.ConnectionClosedOK:
            print("Server has cleanly disconnected us")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    server = os.environ.get("SERVER", "localhost")
    port = os.environ.get("PORT", "8000")
    name = os.environ.get("NAME", getpass.getuser())
    loop.run_until_complete(agent_loop(f"{server}:{port}", name))
