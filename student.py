from __future__ import annotations
import asyncio
import json
import os
import getpass  
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import websockets

from consts import COOL_DOWN

MOVE_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "w": (0, -1),
    "s": (0, 1),
    "a": (-1, 0),
    "d": (1, 0),
}

DIRECTION_TO_DELTA: Dict[int, Tuple[int, int]] = {
    0: (0, -1),  # NORTH
    1: (1, 0),   # EAST
    2: (0, 1),   # SOUTH
    3: (-1, 0),  # WEST
}


@dataclass
class AgentConfig:
    map_width: int = 40
    map_height: int = 24
    shot_score_threshold: float = 28.0
    mushroom_clear_threshold: float = 18.0
    target_band_rows: int = 6
    max_search_rows: int = 24
    max_vertical_travel_up: int = 9
    panic_trigger_distance: int = 8
    panic_escape_depth: int = 8  # Depth explored when plotting escape routes
    defensive_escape_depth: int = 6
    defensive_head_distance: int = 9
    defensive_pressure_threshold: int = 3
    prediction_steps: int = 14
    intercept_tolerance_frames: int = 1  # Allow slight mismatch when predicting hits
    max_path_length: int = 50
    alignment_row_offset: int = 2
    head_prediction_horizon: int = 10
    column_blocker_penalty: float = 0.45
    stuck_blocker_threshold: int = 2  # Reduced to clear paths earlier
    stuck_head_distance_rows: int = 6  # Reduced to be more sensitive to heads
    stuck_clear_bonus: float = 240.0
    head_bonus: float = 230.0
    height_weight: float = 16.0
    same_column_bonus: float = 90.0
    distance_penalty: float = 1.4
    mushroom_penalty: float = 18.0
    mushroom_clear_bonus: float = 30.0
    head_focus_bonus: float = 180.0
    aggressive_vertical_enabled: bool = True
    aggressive_min_score_gain: float = 6.0
    max_stuck_attempts: int = 2  # Reduced to react faster to being stuck
    head_clear_distance: int = 8  # Increased to clear heads from farther
    small_centipede_bonus: float = 200.0
    blocking_mushroom_bonus: float = 220.0
    shot_cooldown_frames: int = COOL_DOWN  # match server cooldown to avoid invalid shots
    panic_wall_penalty: float = 1.2
    safe_space_bonus: float = 0.8
    medium_danger_penalty: float = 12.0
    medium_neighbor_penalty: float = 0.7
    column_threat_horizon: int = 7
    column_dodge_vertical_allowance: int = 3
    near_head_clear_distance: int = 8
    short_term_horizon: int = 4
    medium_term_horizon: int = 12
    target_column_recalc_interval: int = 5
    defense_activation_distance: int = 5
    low_space_threshold: int = 1
    defensive_column_trigger_frames: int = 5
    defensive_threat_penalty: float = 130.0
    max_acceptable_shot_wait: int = 16
    long_shot_penalty: float = 2.6
    intercept_patience_frames: int = 10
    trapped_head_bonus: float = 420.0
    trapped_column_alignment_bonus: float = 6.0
    aggressive_shot_margin: float = 7.5
    trapped_mushroom_bonus: float = 260.0
    safe_band_clear_threshold: int = 4
    safe_band_clear_bonus: float = 200.0
    critical_head_rows: int = 4
    critical_head_bonus: float = 240.0
    edge_bias_weight: float = 1.0
    center_bias_weight: float = 0.8
    preemptive_column_clear_rows: int = 5
    preemptive_column_clear_threshold: int = 2
    critical_escape_space: int = 2

@dataclass
class CentipedeSnapshot:
    name: str
    body: List[Tuple[int, int]]
    direction: int

    @property
    def head(self) -> Tuple[int, int]:
        return self.body[-1]

    @property
    def segments(self) -> List[Tuple[int, int]]:
        return self.body


@dataclass
class GameSnapshot:
    player_pos: Tuple[int, int]
    player_alive: bool
    mushrooms: Dict[Tuple[int, int], int]
    centipedes: List[CentipedeSnapshot]
    centipede_tiles: Set[Tuple[int, int]]
    segment_lookup: Dict[Tuple[int, int], CentipedeSnapshot]


@dataclass
class ColumnProfile:
    first_type: str
    first_pos: Tuple[int, int]
    blockers: int
    nearest_cent: Optional[Tuple[int, int]]
    centipede: Optional[CentipedeSnapshot]
    nearest_is_head: bool
    mushroom_health: Optional[int] = None


@dataclass
class ShotCandidate:
    target: Tuple[int, int]
    score: float
    type: str
    is_head: bool
    frames_until_hit: Optional[int] = None
    confidence: float = 1.0


@dataclass
class AgentContext:
    danger_tiles: Set[Tuple[int, int]]
    fatal_tiles: Set[Tuple[int, int]]
    medium_danger_tiles: Set[Tuple[int, int]]
    column_profile: Optional[ColumnProfile]
    can_hit_now: bool
    shot_candidate: Optional[ShotCandidate]
    target_columns: List[int]
    player_under_direct_threat: bool
    player_under_medium_threat: bool
    current_intercept_frames: Optional[int]
    column_threats: Dict[int, int]
    defensive_mode: bool
    trapped_columns: Dict[int, Tuple[CentipedeSnapshot, Tuple[Tuple[int, int], ...]]]


class StudentAgent:
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.map_width = self.config.map_width
        self.map_height = self.config.map_height
        self._current_plan: Deque[str] = deque()
        self._danger_tiles: Set[Tuple[int, int]] = set()
        self._fatal_tiles: Set[Tuple[int, int]] = set()
        self._medium_danger_tiles: Set[Tuple[int, int]] = set()
        self._column_path_blockers: Dict[int, int] = {}
        self._columns_needing_clear: Set[int] = set()
        self._stuck_counter = 0
        self._last_pos: Optional[Tuple[int, int]] = None
        self._pending_emergency_action: Optional[str] = None
        self._frame = 0
        self._last_shot_frame = -9999
        self._simulation_cache_frame = -1
        self._simulation_cache_steps = max(
            self.config.max_path_length,
            self.config.prediction_steps,
            self.config.medium_term_horizon,
        )
        self._centipede_state_cache: Dict[int, List[List[Tuple[int, int]]]] = {}
        self._centipede_head_cache: Dict[int, List[Tuple[int, int]]] = {}
        self._bfs_cache_frame = -1
        self._bfs_cache: Dict[
            Tuple[Tuple[int, int], Tuple[int, ...], Tuple[int, ...], int],
            List[str],
        ] = {}
        self._target_columns_cache: List[int] = []
        self._target_columns_frame = -1
        self._trapped_columns: Dict[int, Tuple[CentipedeSnapshot, Tuple[Tuple[int, int], ...]]] = {}

    def process_server_update(self, state: Dict[str, Any]) -> Optional[str]:
        if "size" in state:
            self.map_width, self.map_height = state["size"]
            return None

        if "highscores" in state:
            return None

        snapshot = self._parse_state(state)
        if not snapshot or not snapshot.player_alive:
            return ""

        self._frame += 1
        return self._choose_command(snapshot)

    def _choose_command(self, state: GameSnapshot) -> str:
        self._prepare_frame_caches()
        context = self._build_context(state)

        self._detect_and_resolve_stuck(state)
        vertical_congested = self._vertical_path_congested(state)

        if self._pending_emergency_action:
            action = self._pending_emergency_action
            if action == "shoot":
                if self._shot_ready():
                    self._pending_emergency_action = None
                    self._record_shot()
                    return "A"
            elif action in ("a", "d", "w", "s"):
                self._pending_emergency_action = None
                return action
            # keep pending until it is safe to execute
            return ""

        if self._immediate_threat(state):
            if self._threat_shootable(state):
                self._record_shot()
                return "A"
            escape_move = self._emergency_evade(state)
            if escape_move:
                return escape_move
            self._record_shot()
            return "A"

        if context.can_hit_now and self._shot_ready():
            if self._should_fire(vertical_congested, context, None, True):
                self._record_shot()
                return "A"

        if context.shot_candidate and self._shot_ready():
            if self._should_fire(
                vertical_congested, context, context.shot_candidate, context.can_hit_now
            ):
                self._record_shot()
                return "A"

        move = self._next_move(state, context)
        return move or ""

    def _prepare_frame_caches(self) -> None:
        if self._simulation_cache_frame != self._frame:
            self._simulation_cache_frame = self._frame
            self._centipede_state_cache.clear()
            self._centipede_head_cache.clear()
        if self._bfs_cache_frame != self._frame:
            self._bfs_cache_frame = self._frame
            self._bfs_cache.clear()

    def _vertical_path_congested(self, state: GameSnapshot) -> bool:
        column = state.player_pos[0]
        blockers = self._path_blocker_count(column, state.player_pos[1], state.mushrooms)
        return blockers >= 2

    def _should_fire(
        self,
        vertical_congested: bool,
        context: AgentContext,
        candidate: Optional[ShotCandidate],
        can_hit_now: bool,
    ) -> bool:
        if context:
            if context.player_under_direct_threat:
                return False
            if context.player_under_medium_threat and not (
                candidate and candidate.is_head
            ):
                return False
        patience_limit = max(1, self.config.max_acceptable_shot_wait)
        frames_until_hit = candidate.frames_until_hit if candidate else None
        if frames_until_hit is None:
            frames_until_hit = context.current_intercept_frames
        candidate_score = candidate.score if candidate else 0.0
        high_value_shot = candidate_score >= (
            self.config.shot_score_threshold + self.config.aggressive_shot_margin
        )
        candidate_is_head = candidate.is_head if candidate else False
        target_column: Optional[int] = None
        if candidate and candidate.target:
            target_column = candidate.target[0]
        elif context.column_profile and context.column_profile.nearest_cent:
            target_column = context.column_profile.nearest_cent[0]
        trapped_target = (
            target_column is not None and target_column in self._trapped_columns
        )
        if (
            frames_until_hit is not None
            and not vertical_congested
            and frames_until_hit > patience_limit
            and not (candidate_is_head or high_value_shot or trapped_target)
        ):
            return False
        defensive_patience = max(1, self.config.intercept_patience_frames)
        if (
            context
            and context.defensive_mode
            and frames_until_hit is not None
            and frames_until_hit > defensive_patience
            and not (candidate and candidate.type == "mushroom" and trapped_target)
            and not candidate_is_head
            and not high_value_shot
        ):
            return False
        if not vertical_congested:
            return True
        if candidate:
            if candidate.type == "mushroom":
                return True
            if candidate.is_head:
                return True
            if trapped_target:
                return True
            if high_value_shot:
                return True
        if can_hit_now:
            return True
        profile = context.column_profile
        if profile and profile.nearest_is_head:
            return True
        return False

    def _build_context(self, state: GameSnapshot) -> AgentContext:
        fatal_tiles, danger_tiles, medium_tiles = self._compute_threat_tiles(state)
        self._danger_tiles = danger_tiles
        self._fatal_tiles = fatal_tiles
        self._medium_danger_tiles = medium_tiles
        column_profile = self._column_profile(state)
        incoming_threats = self._incoming_column_threats(state)
        trapped_columns = self._detect_trapped_columns(state)
        self._trapped_columns = trapped_columns
        intercept_match = self._find_intercept_match(
            state.player_pos[0], state.player_pos[1], state
        )
        can_hit_now = intercept_match is not None
        current_intercept_frames = intercept_match[0] if intercept_match else None
        target_columns = self._select_target_columns(
            state,
            incoming_threats=incoming_threats,
            trapped_columns=trapped_columns,
        )
        shot_candidate = self._select_shot_candidate(
            state, profile=column_profile, can_hit_now=can_hit_now
        )
        player_under_direct = (
            state.player_pos in danger_tiles or state.player_pos in fatal_tiles
        )
        player_under_medium = state.player_pos in medium_tiles
        low_space = self._low_space(state)
        nearest_head_distance = self._closest_head_distance(state)
        defensive_mode = player_under_direct or player_under_medium or low_space
        if nearest_head_distance is not None:
            defensive_mode = defensive_mode or (
                nearest_head_distance <= self.config.defense_activation_distance
            )
        return AgentContext(
            danger_tiles=danger_tiles,
            fatal_tiles=fatal_tiles,
            medium_danger_tiles=medium_tiles,
            column_profile=column_profile,
            can_hit_now=can_hit_now,
            shot_candidate=shot_candidate,
            target_columns=target_columns,
            player_under_direct_threat=player_under_direct,
            player_under_medium_threat=player_under_medium,
            current_intercept_frames=current_intercept_frames,
            column_threats=incoming_threats,
            defensive_mode=defensive_mode,
            trapped_columns=trapped_columns,
        )

    def _detect_and_resolve_stuck(self, state: GameSnapshot) -> None:
        if self._last_pos == state.player_pos:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_pos = state.player_pos

        if self._stuck_counter < self.config.max_stuck_attempts:
            return

        self._stuck_counter = 0
        px, py = state.player_pos
        mush = state.mushrooms
        profile = self._column_profile(state)

        if profile and profile.first_type == "mushroom":
            if profile.blockers >= self.config.stuck_blocker_threshold or (
                (px, py - 1) == profile.first_pos
            ):
                self._pending_emergency_action = "shoot"
                return

        if (px, py - 1) in mush:
            self._pending_emergency_action = "shoot"
            return

        lateral_preferences: List[str] = []
        for move in ("a", "d"):
            nxt = self._apply_move(state.player_pos, move)
            if not self._in_bounds(nxt) or nxt in state.mushrooms:
                continue
            if nxt in self._fatal_tiles:
                continue
            if nxt in self._danger_tiles:
                lateral_preferences.append(move)
                continue
            self._current_plan.clear()
            self._current_plan.append(move)
            return

        for move in ("w", "s"):
            nxt = self._apply_move(state.player_pos, move)
            if not self._in_bounds(nxt) or nxt in state.mushrooms:
                continue
            if nxt in self._fatal_tiles or nxt in self._danger_tiles:
                continue
            self._current_plan.clear()
            self._current_plan.append(move)
            return

        if lateral_preferences:
            self._current_plan.clear()
            self._current_plan.append(lateral_preferences[0])
            return

        for move in ("w", "s"):
            nxt = self._apply_move(state.player_pos, move)
            if self._in_bounds(nxt) and nxt not in state.mushrooms:
                self._current_plan.clear()
                self._current_plan.append(move)
                return

        self._current_plan.clear()
        self._current_plan.append("a")

    def _immediate_threat(self, state: GameSnapshot) -> bool:
        px, py = state.player_pos
        if (px, py) in state.centipede_tiles:
            return True
        for delta in MOVE_TO_DELTA.values():
            neighbor = (px + delta[0], py + delta[1])
            if neighbor in state.centipede_tiles:
                return True
        return False

    def _threat_shootable(self, state: GameSnapshot) -> bool:
        px, py = state.player_pos
        for y in range(py - 1, -1, -1):
            pos = (px, y)
            if pos in state.mushrooms:
                return False
            if pos in state.centipede_tiles:
                return True
        return False

    def _emergency_evade(self, state: GameSnapshot) -> str:
        priorities = self._threat_priorities(state)
        for allow_danger in (False, True):
            for move in priorities:
                nxt = self._apply_move(state.player_pos, move)
                if not self._in_bounds(nxt):
                    continue
                if nxt in state.mushrooms:
                    continue
                if nxt in self._fatal_tiles:
                    continue
                if not allow_danger and (
                    nxt in self._danger_tiles or nxt in self._medium_danger_tiles
                ):
                    continue
                return move
        return ""

    def _threat_priorities(self, state: GameSnapshot) -> List[str]:
        px, py = state.player_pos
        threat = {"d": False, "a": False, "s": False, "w": False}
        for cx, cy in state.centipede_tiles:
            if cy == py:
                if 0 < px - cx <= 3:
                    threat["d"] = True
                if 0 < cx - px <= 3:
                    threat["a"] = True
            if cx == px:
                if 0 < py - cy <= 2:
                    threat["s"] = True
                if 0 < cy - py <= 2:
                    threat["w"] = True
        ordered: List[str] = [move for move, flagged in threat.items() if flagged]
        for move in ("d", "a", "w", "s"):
            if move not in ordered:
                ordered.append(move)
        return ordered

    def _closest_head_distance(self, state: GameSnapshot) -> Optional[int]:
        best_dist: Optional[int] = None
        px, py = state.player_pos
        for cent in state.centipedes:
            hx, hy = cent.head
            dist = abs(hx - px) + abs(hy - py)
            if best_dist is None or dist < best_dist:
                best_dist = dist
        return best_dist

    def _next_move(self, state: GameSnapshot, context: Optional[AgentContext] = None) -> str:
        if self._current_plan and not self._plan_still_valid(state):
            self._current_plan.clear()

        if not self._current_plan:
            new_plan = self._build_plan(state, context)
            self._current_plan.extend(new_plan)

        if self._current_plan:
            next_move = self._current_plan.popleft()
            return next_move

        return ""

    def _build_plan(
        self, state: GameSnapshot, context: Optional[AgentContext] = None
    ) -> Sequence[str]:
        if (
            state.centipede_tiles
            and self._min_distance(state.player_pos, state.centipede_tiles)
            <= self.config.panic_trigger_distance
        ):
            escape_plan = self._panic_escape_plan(state)
            if escape_plan:
                return escape_plan
            panic_move = self._panic_move(state)
            if panic_move:
                return [panic_move]

        if context and self._should_defensive_retreat(state, context):
            defensive_plan = self._defensive_escape_plan(state)
            if defensive_plan:
                return defensive_plan

        if context and self._should_relocate_from_threat(state, context):
            relocation_plan = self._relocate_from_threat(state, context)
            if relocation_plan:
                return relocation_plan

        column_threat = self._column_threat_detected(state)
        if column_threat is not None:
            dodge_plan = self._column_dodge_plan(state, column_threat)
            if dodge_plan:
                return dodge_plan

        safe_band_plan = self._safe_band_sweeping_plan(state, context)
        if safe_band_plan:
            return safe_band_plan

        center_plan = self._center_reposition_plan(state, context)
        if center_plan:
            return center_plan

        bottom_zone_threshold = self.map_height - 3
        any_low = any(cent.head[1] >= bottom_zone_threshold for cent in state.centipedes)
        px, py = state.player_pos

        preferred_row = self._preferred_row(state, any_low)
        if py != preferred_row:
            moves: List[str] = []
            direction = "w" if py > preferred_row else "s"
            current_y = py
            while current_y != preferred_row:
                next_pos = self._apply_move((px, current_y), direction)
                if not self._in_bounds(next_pos):
                    moves = []
                    break
                if next_pos in state.mushrooms or next_pos in self._fatal_tiles:
                    moves = []
                    break
                moves.append(direction)
                current_y = next_pos[1]
            if moves:
                return moves

        can_hit_now = context.can_hit_now if context else self._can_hit_now(state)
        if can_hit_now:
            return []

        intercept_plan = self._find_intercept_plan(state)
        if intercept_plan:
            return intercept_plan

        if (
            self.config.aggressive_vertical_enabled
            and context
            and self._should_aggressive_ascent(state, context)
        ):
            ascent_plan = self._try_aggressive_ascent(state)
            if ascent_plan:
                return ascent_plan

        target_columns = (
            context.target_columns if context else self._select_target_columns(state)
        )
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        base_min_row = max(0, self.map_height - self.config.max_search_rows)
        extended_min_row = max(0, state.player_pos[1] - self.config.max_vertical_travel_up)
        min_row = min(base_min_row, extended_min_row)
        allowed_rows = set(range(min_row, self.map_height))
        safe_rows = set(range(safe_row_start, self.map_height))

        if self._should_descend(state):
            descend_path = self._bfs_to_column(
                start=state.player_pos,
                target_columns=list(range(self.map_width)),
                allowed_rows=safe_rows,
                blocked=blocked,
            )
            if descend_path:
                return descend_path

        clear_column = self._choose_clear_column(state)
        if clear_column is not None:
            clear_plan = self._alignment_plan(state, clear_column)
            if clear_plan is not None:
                return clear_plan

        viable_columns = [
            col for col in target_columns if self._column_path_blockers.get(col, 0) <= 1
        ]
        primary_column = (
            viable_columns[0]
            if viable_columns
            else (target_columns[0] if target_columns else None)
        )
        if primary_column is not None:
            alignment_plan = self._alignment_plan(state, primary_column)
            if alignment_plan:
                return alignment_plan

        path = self._bfs_to_column(
            start=state.player_pos,
            target_columns=target_columns,
            allowed_rows=set(allowed_rows),
            blocked=blocked,
        )

        if path:
            return path

        fallback = self._best_local_move(state)
        return [fallback] if fallback else []

    def _should_aggressive_ascent(
        self, state: GameSnapshot, context: AgentContext
    ) -> bool:
        if context.player_under_direct_threat or context.player_under_medium_threat:
            return False
        if context.can_hit_now:
            return False
        if not state.centipedes:
            return False
        head_pos = self._nearest_head_position(state)
        if not head_pos:
            return False
        px, py = state.player_pos
        head_y = head_pos[1]
        if head_y >= py - 1:
            return False
        if (py - head_y) < 3:
            return False
        if (
            self._path_blocker_count(px, py, state.mushrooms)
            >= self.config.stuck_blocker_threshold
        ):
            return False
        return True

    def _plan_still_valid(self, state: GameSnapshot) -> bool:
        if not self._current_plan:
            return False
        if self._head_threatening_column(state):
            return False

        next_move = self._current_plan[0]
        next_pos = self._apply_move(state.player_pos, next_move)
        if not self._in_bounds(next_pos):
            return False
        if next_pos in state.mushrooms or next_pos in self._danger_tiles:
            return False
        if (
            next_pos in self._medium_danger_tiles
            and len(self._current_plan) > 1
        ):
            return False
        return True

    def _head_threatening_column(self, state: GameSnapshot) -> bool:
        px, py = state.player_pos
        for cent in state.centipedes:
            hx, hy = cent.head
            if hx != px:
                continue
            if abs(hy - py) <= self.config.near_head_clear_distance:
                return True
        return False

    def _bfs_to_column(
        self,
        start: Tuple[int, int],
        target_columns: Sequence[int],
        allowed_rows: Set[int],
        blocked: Set[Tuple[int, int]],
    ) -> List[str]:
        if not target_columns:
            return []

        cache_key = (
            start,
            tuple(sorted(set(target_columns))),
            tuple(sorted(allowed_rows)),
            id(blocked),
        )
        if cache_key in self._bfs_cache:
            return list(self._bfs_cache[cache_key])

        queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque([(start, [])])
        visited: Set[Tuple[int, int]] = {start}

        while queue:
            pos, path = queue.popleft()
            if len(path) > self.config.max_path_length:
                continue

            if pos[0] in target_columns and pos[1] in allowed_rows:
                self._bfs_cache[cache_key] = list(path)
                return path

            for move, delta in MOVE_TO_DELTA.items():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in visited or nxt in blocked:
                    continue
                next_step = len(path) + 1
                if nxt in self._danger_tiles and len(path) > 0:
                    continue
                if (
                    nxt in self._medium_danger_tiles
                    and next_step >= self.config.short_term_horizon
                ):
                    continue
                if not self._in_bounds(nxt):
                    continue

                visited.add(nxt)
                queue.append((nxt, path + [move]))

        self._bfs_cache[cache_key] = []
        return []

    def _best_local_move(self, state: GameSnapshot) -> str:
        best_move = ""
        best_score = float("-inf")
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)

        for move, delta in MOVE_TO_DELTA.items():
            nxt = (state.player_pos[0] + delta[0], state.player_pos[1] + delta[1])
            if not self._in_bounds(nxt):
                continue
            if nxt in state.mushrooms or nxt in self._danger_tiles:
                continue
            medium_penalty = (
                self.config.medium_danger_penalty
                if nxt in self._medium_danger_tiles
                else 0.0
            )

            distance_penalty = self._min_distance(nxt, state.centipede_tiles)
            score = distance_penalty * 3.0
            score -= abs(nxt[0] - (self.map_width // 2)) * self.config.center_bias_weight
            score -= max(
                0, (self.map_height - self.config.target_band_rows) - nxt[1]
            ) * 0.2
            if state.player_pos[1] < safe_row_start:
                if move == "s":
                    score += 4.0
                elif move == "w":
                    score -= 6.0
            score -= medium_penalty
            edge_distance = min(nxt[0], self.map_width - 1 - nxt[0])
            if edge_distance < 3:
                score -= (3 - edge_distance) * self.config.edge_bias_weight

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _alignment_plan(
        self, state: GameSnapshot, target_column: int
    ) -> Optional[List[str]]:
        current_x, current_y = state.player_pos
        alignment_row = self._alignment_row()
        plan: List[str] = []

        if target_column == current_x and current_y == alignment_row:
            return []

        while current_y != alignment_row:
            direction = "s" if current_y < alignment_row else "w"
            next_pos = self._apply_move((current_x, current_y), direction)
            if not self._in_bounds(next_pos):
                return None
            if (
                next_pos in state.mushrooms
                or next_pos in self._fatal_tiles
                or next_pos in self._danger_tiles
                or next_pos in self._medium_danger_tiles
            ):
                return None
            plan.append(direction)
            current_x, current_y = next_pos

        dx = target_column - current_x
        if dx == 0:
            return plan
        step = 1 if dx > 0 else -1
        move_key = "d" if dx > 0 else "a"
        for _ in range(abs(dx)):
            next_x = current_x + step
            next_pos = (next_x, current_y)
            if not self._in_bounds(next_pos):
                return None
            if (
                next_pos in state.mushrooms
                or next_pos in self._fatal_tiles
                or next_pos in self._danger_tiles
                or next_pos in self._medium_danger_tiles
            ):
                return None
            plan.append(move_key)
            current_x = next_x

        return plan

    def _choose_clear_column(self, state: GameSnapshot) -> Optional[int]:
        if not self._columns_needing_clear:
            return None

        player_x = state.player_pos[0]
        best_column: Optional[int] = None
        best_score = float("-inf")

        for column in self._columns_needing_clear:
            distance = abs(player_x - column)
            blockers = self._column_path_blockers.get(column, 0)
            score = -distance - blockers * 0.5
            if column in self._trapped_columns:
                score += self.config.trapped_column_alignment_bonus
            if score > best_score:
                best_score = score
                best_column = column

        return best_column

    def _column_threat_detected(self, state: GameSnapshot) -> Optional[str]:
        px, py = state.player_pos
        horizon = max(1, self.config.column_threat_horizon)
        for cent in state.centipedes:
            preferred = self._preferred_dodge_direction(cent, px)
            predicted = self._simulate_centipede_future_positions(
                cent, state.mushrooms, horizon
            )
            for step, (cx, cy) in enumerate(predicted, start=1):
                if cx != px:
                    continue
                if cy > py:
                    continue
                if (py - cy) > self.config.near_head_clear_distance:
                    continue
                if self._can_intercept_threat(px, py, cy, state, cent, step):
                    continue
                return preferred
        return None

    def _should_relocate_from_threat(
        self, state: GameSnapshot, context: AgentContext
    ) -> bool:
        if not context.column_threats:
            return False
        column = state.player_pos[0]
        threat = context.column_threats.get(column)
        if threat is None:
            return False
        trigger = max(1, self.config.defensive_column_trigger_frames)
        if context.can_hit_now:
            return False
        if context.shot_candidate and context.shot_candidate.is_head:
            return False
        if threat <= trigger:
            return True
        if context.defensive_mode and threat <= (trigger + 2):
            return True
        return False

    def _relocate_from_threat(
        self, state: GameSnapshot, context: AgentContext
    ) -> Optional[List[str]]:
        trigger = max(1, self.config.defensive_column_trigger_frames)
        safe_columns: List[int] = []
        for col in range(self.map_width):
            if col == state.player_pos[0]:
                continue
            threat = context.column_threats.get(col)
            if threat is not None and threat <= trigger:
                continue
            blockers = self._column_path_blockers.get(col, 0)
            if blockers >= (self.config.stuck_blocker_threshold + 1):
                continue
            safe_columns.append(col)

        if not safe_columns:
            return None

        min_row = max(
            0, state.player_pos[1] - self.config.column_dodge_vertical_allowance
        )
        max_row = min(
            self.map_height,
            state.player_pos[1] + self.config.column_dodge_vertical_allowance + 1,
        )
        allowed_rows = set(range(min_row, max_row))
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        path = self._bfs_to_column(
            start=state.player_pos,
            target_columns=safe_columns,
            allowed_rows=allowed_rows,
            blocked=blocked,
        )
        return path

    def _can_intercept_threat(
        self,
        column: int,
        from_y: int,
        target_y: int,
        state: GameSnapshot,
        cent: CentipedeSnapshot,
        frames: int,
    ) -> bool:
        if not self._shot_ready():
            return False
        if not self._path_clear_for_shot(column, from_y, target_y, state.mushrooms):
            return False
        return self._intercept_window_exists(
            column,
            from_y,
            state,
            arrival_delay=0,
            target_centipede=cent,
            max_frames=frames,
        )

    def _preferred_dodge_direction(
        self, cent: CentipedeSnapshot, player_x: int
    ) -> Optional[str]:
        head_x = cent.head[0]
        if head_x < player_x:
            return "d"
        if head_x > player_x:
            return "a"
        if cent.direction == 1:
            return "d"
        if cent.direction == 3:
            return "a"
        return None

    def _column_dodge_plan(
        self, state: GameSnapshot, preferred_move: Optional[str]
    ) -> Optional[List[str]]:
        move = self._select_safe_lateral_move(
            state, preferred_move=preferred_move, allow_medium=False
        )
        if move:
            return [move]
        move = self._select_safe_lateral_move(
            state, preferred_move=preferred_move, allow_medium=True
        )
        if move:
            return [move]

        target_columns = [col for col in range(self.map_width) if col != state.player_pos[0]]
        if not target_columns:
            return None

        min_row = max(
            0, state.player_pos[1] - self.config.column_dodge_vertical_allowance
        )
        max_row = min(
            self.map_height,
            state.player_pos[1] + self.config.column_dodge_vertical_allowance + 1,
        )
        allowed_rows = set(range(min_row, max_row))
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        path = self._bfs_to_column(
            start=state.player_pos,
            target_columns=target_columns,
            allowed_rows=allowed_rows,
            blocked=blocked,
        )
        if path:
            return path
        return None

    def _safe_band_sweeping_plan(
        self, state: GameSnapshot, context: Optional[AgentContext]
    ) -> Optional[List[str]]:
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        critical_row = max(0, safe_row_start - self.config.critical_head_rows)
        candidate_heads: List[Tuple[int, int]] = [
            cent.head
            for cent in state.centipedes
            if cent.head[1] >= critical_row
        ]
        candidate_heads.sort(
            key=lambda pos: (pos[1], abs(pos[0] - state.player_pos[0]))
        )
        for head in candidate_heads:
            plan = self._alignment_plan(state, head[0])
            if plan:
                return plan

        safe_band_counts = self._safe_band_mushroom_counts(state)
        if not safe_band_counts:
            return None
        max_count = max(safe_band_counts.values())
        if max_count < self.config.safe_band_clear_threshold:
            return None
        target_columns = [
            col for col, count in safe_band_counts.items() if count == max_count
        ]
        target_columns.sort(key=lambda col: abs(col - state.player_pos[0]))
        for column in target_columns:
            plan = self._alignment_plan(state, column)
            if plan:
                return plan
        allowed_rows = set(range(max(0, safe_row_start - 2), self.map_height))
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        path = self._bfs_to_column(
            start=state.player_pos,
            target_columns=target_columns,
            allowed_rows=allowed_rows,
            blocked=blocked,
        )
        return path

    def _center_reposition_plan(
        self, state: GameSnapshot, context: Optional[AgentContext]
    ) -> Optional[List[str]]:
        edge_distance = min(state.player_pos[0], self.map_width - 1 - state.player_pos[0])
        low_space = self._low_space(state)
        if (
            edge_distance > self.config.critical_escape_space
            and not low_space
            and not (context and context.player_under_medium_threat)
        ):
            return None
        preferred_centers = [
            self.map_width // 2,
            (self.map_width // 2) - 1,
            (self.map_width // 2) + 1,
        ]
        for column in preferred_centers:
            if column < 0 or column >= self.map_width:
                continue
            plan = self._alignment_plan(state, column)
            if plan is not None:
                return plan
        allowed_rows = set(
            range(
                max(0, state.player_pos[1] - 2),
                min(self.map_height, state.player_pos[1] + 3),
            )
        )
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        path = self._bfs_to_column(
            start=state.player_pos,
            target_columns=[self.map_width // 2],
            allowed_rows=allowed_rows,
            blocked=blocked,
        )
        return path

    def _select_safe_lateral_move(
        self,
        state: GameSnapshot,
        preferred_move: Optional[str] = None,
        allow_medium: bool = False,
    ) -> Optional[str]:
        order: List[str] = []
        if preferred_move:
            order.append(preferred_move)
            if preferred_move == "a":
                order.append("d")
            elif preferred_move == "d":
                order.append("a")
        if not order:
            order = ["d", "a"]

        best_move: Optional[str] = None
        best_score = float("-inf")
        for move in order:
            nxt = self._apply_move(state.player_pos, move)
            if not self._is_tile_safe(nxt, state, allow_medium=allow_medium):
                continue
            score = self._min_distance(nxt, state.centipede_tiles)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _panic_move(self, state: GameSnapshot) -> str:
        best_safe_move = ""
        best_safe_score = float("-inf")
        best_risky_move = ""
        best_risky_score = float("-inf")

        for move, delta in MOVE_TO_DELTA.items():
            nxt = (state.player_pos[0] + delta[0], state.player_pos[1] + delta[1])
            if not self._in_bounds(nxt):
                continue
            if (
                nxt in state.mushrooms
                or nxt in self._fatal_tiles
                or nxt in self._danger_tiles
            ):
                continue

            distance = self._min_distance(nxt, state.centipede_tiles)
            score = distance
            up = self._apply_move(nxt, "w")
            if (
                up
                and self._in_bounds(up)
                and up not in state.mushrooms
                and up not in self._fatal_tiles
            ):
                score += 2.0
            if nxt in self._medium_danger_tiles:
                score -= self.config.medium_danger_penalty * 0.5
                if score > best_risky_score:
                    best_risky_score = score
                    best_risky_move = move
            else:
                if score > best_safe_score:
                    best_safe_score = score
                    best_safe_move = move

        if best_safe_move:
            return best_safe_move
        return best_risky_move

    def _should_defensive_retreat(
        self, state: GameSnapshot, context: Optional[AgentContext]
    ) -> bool:
        if not state.centipedes:
            return False
        px, py = state.player_pos
        if state.player_pos in self._danger_tiles:
            return True
        head_threat = False
        for cent in state.centipedes:
            if (
                abs(cent.head[0] - px) + abs(cent.head[1] - py)
                <= self.config.defensive_head_distance
            ):
                head_threat = True
                break
        medium_tile = state.player_pos in self._medium_danger_tiles
        if medium_tile and head_threat:
            return True
        pressure = 0
        for delta in MOVE_TO_DELTA.values():
            neighbor = (px + delta[0], py + delta[1])
            if neighbor in self._danger_tiles:
                pressure += 2
            elif neighbor in self._medium_danger_tiles:
                pressure += 1
        if head_threat and pressure >= self.config.defensive_pressure_threshold:
            return True
        column_blocked = (
            self._path_blocker_count(px, py, state.mushrooms)
            >= self.config.stuck_blocker_threshold
        )
        if head_threat and column_blocked and (
            medium_tile or (context and not context.can_hit_now)
        ):
            return True
        return False

    def _panic_escape_plan(self, state: GameSnapshot) -> Optional[List[str]]:
        depth_limit = max(1, self.config.panic_escape_depth)
        start = state.player_pos
        queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque([(start, [])])
        visited: Set[Tuple[int, int]] = {start}
        best_path: Optional[List[str]] = None
        best_score = float("-inf")

        while queue:
            pos, path = queue.popleft()
            score = self._panic_position_score(pos, state)
            if path and score > best_score:
                best_score = score
                best_path = path

            if len(path) >= depth_limit:
                continue

            for move, delta in MOVE_TO_DELTA.items():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if not self._in_bounds(nxt):
                    continue
                if nxt in state.mushrooms or nxt in self._fatal_tiles:
                    continue
                if nxt in self._danger_tiles:
                    continue
                next_step = len(path) + 1
                if (
                    nxt in self._medium_danger_tiles
                    and next_step >= self.config.short_term_horizon
                ):
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [move]))

        return best_path

    def _defensive_escape_plan(self, state: GameSnapshot) -> Optional[List[str]]:
        depth_limit = max(1, self.config.defensive_escape_depth)
        start = state.player_pos
        queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque([(start, [])])
        visited: Set[Tuple[int, int]] = {start}
        best_path: Optional[List[str]] = None
        best_score = float("-inf")

        while queue:
            pos, path = queue.popleft()
            score = self._panic_position_score(pos, state)
            if path and score > best_score:
                best_score = score
                best_path = path

            if len(path) >= depth_limit:
                continue

            for move, delta in MOVE_TO_DELTA.items():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if not self._in_bounds(nxt):
                    continue
                if nxt in state.mushrooms or nxt in self._fatal_tiles:
                    continue
                if nxt in self._danger_tiles:
                    continue
                if len(path) == 0 and nxt in self._medium_danger_tiles:
                    continue
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [move]))

        return best_path

    def _panic_position_score(self, pos: Tuple[int, int], state: GameSnapshot) -> float:
        distance = self._min_distance(pos, state.centipede_tiles)
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        row_bonus = max(0, pos[1] - safe_row_start) * 0.6
        neighbor_penalty = 0.0
        open_tiles = 0
        medium_penalty = (
            self.config.medium_danger_penalty if pos in self._medium_danger_tiles else 0.0
        )
        for delta in MOVE_TO_DELTA.values():
            neighbor = (pos[0] + delta[0], pos[1] + delta[1])
            if neighbor in state.centipede_tiles or neighbor in self._danger_tiles:
                neighbor_penalty += 1.5
            elif neighbor in self._medium_danger_tiles:
                neighbor_penalty += self.config.medium_neighbor_penalty
            elif neighbor in state.mushrooms:
                neighbor_penalty += 0.3
            else:
                open_tiles += 1

        edge_distance = min(pos[0], self.map_width - 1 - pos[0])
        edge_penalty = max(0, (2 - edge_distance)) * self.config.panic_wall_penalty
        space_bonus = open_tiles * self.config.safe_space_bonus

        return (
            distance * 3.0
            + row_bonus
            + space_bonus
            - neighbor_penalty
            - edge_penalty
            - medium_penalty
        )

    def _select_shot_candidate(
        self,
        state: GameSnapshot,
        profile: Optional[ColumnProfile] = None,
        can_hit_now: bool = False,
    ) -> Optional[ShotCandidate]:
        max_segments = max((len(c.segments) for c in state.centipedes), default=1)
        profile = profile or self._column_profile(state)
        if not profile or not profile.nearest_cent:
            return None

        if profile.first_type == "centipede":
            if not profile.nearest_cent:
                return None
            if not profile.centipede:
                return None
            max_frames = min(self.config.prediction_steps, state.player_pos[1])
            intercept = self._find_intercept_match(
                state.player_pos[0],
                state.player_pos[1],
                state,
                target_centipede=profile.centipede,
                max_frames=max_frames,
            )
            if not intercept:
                mushroom_target = self._blocking_mushroom_target(
                    profile.centipede, state
                )
                if mushroom_target:
                    return mushroom_target
                return None
            score = self._score_centipede_shot(
                profile.nearest_cent, profile.nearest_is_head, state.player_pos[1]
            )
            if profile.centipede:
                score += self._small_centipede_focus(
                    len(profile.centipede.segments), max_segments
                )
            target_column = profile.nearest_cent[0]
            if target_column in self._trapped_columns:
                score += self.config.trapped_head_bonus * 0.35
            frames_until_hit = intercept[0]
            score -= frames_until_hit * self.config.long_shot_penalty
            confidence = 1.0
            if self.config.max_acceptable_shot_wait > 0:
                confidence = max(
                    0.2,
                    1.0
                    - (
                        frames_until_hit
                        / max(1.0, float(self.config.max_acceptable_shot_wait))
                    ),
                )
            if score >= self.config.shot_score_threshold:
                return ShotCandidate(
                    target=profile.nearest_cent,
                    score=score,
                    type="centipede",
                    is_head=profile.nearest_is_head,
                    frames_until_hit=frames_until_hit,
                    confidence=confidence,
                )
        elif profile.first_type == "mushroom":
            column = state.player_pos[0]
            force_clear = column in self._columns_needing_clear or (
                profile.blockers >= self.config.stuck_blocker_threshold
                and profile.nearest_cent is not None
            )
            if column in self._trapped_columns:
                force_clear = True
            if profile.nearest_cent:
                head_distance = abs(profile.nearest_cent[1] - state.player_pos[1])
                if head_distance <= self.config.near_head_clear_distance:
                    force_clear = True
            if not profile.nearest_cent:
                return None

            score = self._score_mushroom_clear(profile, trapped=force_clear)
            if force_clear or (score >= self.config.mushroom_clear_threshold and not can_hit_now):
                frames_until_hit = (
                    state.player_pos[1] - profile.first_pos[1]
                    if profile.first_pos
                    else None
                )
                return ShotCandidate(
                    target=profile.first_pos,
                    score=score,
                    type="mushroom",
                    is_head=profile.nearest_is_head,
                    frames_until_hit=frames_until_hit,
                )

        return None
        preemptive = self._preemptive_column_clear_target(state)
        if preemptive:
            return preemptive

    def _column_profile(self, state: GameSnapshot) -> Optional[ColumnProfile]:
        player_x, player_y = state.player_pos
        first_type: Optional[str] = None
        first_pos: Optional[Tuple[int, int]] = None
        blockers = 0
        mushroom_health: Optional[int] = None
        nearest_cent: Optional[Tuple[int, int]] = None
        centipede: Optional[CentipedeSnapshot] = None
        nearest_is_head = False

        for ty in range(player_y - 1, -1, -1):
            pos = (player_x, ty)
            if pos in state.mushrooms:
                blockers += 1
                if first_type is None:
                    first_type = "mushroom"
                    first_pos = pos
                    mushroom_health = state.mushrooms[pos]
                continue

            if pos in state.centipede_tiles:
                nearest_cent = pos
                centipede = state.segment_lookup.get(pos)
                nearest_is_head = centipede.head == pos if centipede else False
                if first_type is None:
                    first_type = "centipede"
                    first_pos = pos
                break

        if first_type is None or first_pos is None:
            return None

        return ColumnProfile(
            first_type=first_type,
            first_pos=first_pos,
            blockers=blockers,
            nearest_cent=nearest_cent,
            centipede=centipede,
            nearest_is_head=nearest_is_head,
            mushroom_health=mushroom_health,
        )

    def _score_centipede_shot(
        self, segment: Tuple[int, int], is_head: bool, player_y: int
    ) -> float:
        seg_y = segment[1]
        distance = player_y - seg_y
        score = (
            (self.map_height - seg_y) * self.config.height_weight
            + self.config.same_column_bonus
            - distance * self.config.distance_penalty
        )
        if is_head:
            score += self.config.head_bonus
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        if seg_y >= max(0, safe_row_start - self.config.critical_head_rows):
            score += self.config.critical_head_bonus
        return score

    def _score_mushroom_clear(
        self, profile: ColumnProfile, trapped: bool = False
    ) -> float:
        if not profile.nearest_cent:
            return float("-inf")

        cent_y = profile.nearest_cent[1]
        cent_value = (self.map_height - cent_y) * (self.config.height_weight * 0.6)
        if profile.nearest_is_head:
            cent_value += self.config.head_focus_bonus

        mushroom_health = (
            profile.mushroom_health if profile.mushroom_health is not None else 4
        )

        score = (
            self.config.mushroom_clear_bonus
            + cent_value
            - profile.blockers * (self.config.mushroom_penalty * 0.4)
            - mushroom_health * 2.0
        )
        if trapped:
            score += self.config.stuck_clear_bonus

        if (
            profile.first_pos
            and profile.first_pos[0] in self._trapped_columns
        ):
            score += self.config.trapped_mushroom_bonus
        if profile.first_pos and profile.first_pos[1] >= max(
            0, self.map_height - self.config.target_band_rows
        ):
            score += self.config.safe_band_clear_bonus

        return score

    def _select_target_columns(
        self,
        state: GameSnapshot,
        incoming_threats: Optional[Dict[int, int]] = None,
        trapped_columns: Optional[
            Dict[int, Tuple[CentipedeSnapshot, Tuple[Tuple[int, int], ...]]]
        ] = None,
    ) -> List[int]:
        trapped_columns = (
            trapped_columns if trapped_columns is not None else self._detect_trapped_columns(state)
        )
        self._trapped_columns = trapped_columns
        use_cache = (
            not trapped_columns
            and self._target_columns_cache
            and (self._frame - self._target_columns_frame)
            < self.config.target_column_recalc_interval
            and state.centipedes
        )
        if use_cache:
            return list(self._target_columns_cache)

        def threat_penalty(column: int) -> float:
            if not incoming_threats:
                return 0.0
            frames = incoming_threats.get(column)
            if frames is None:
                return 0.0
            horizon = max(1, self.config.column_threat_horizon)
            urgency = max(0, horizon - frames + 1) / horizon
            return self.config.defensive_threat_penalty * urgency

        column_scores: Dict[int, float] = {}
        player_x = state.player_pos[0]
        column_blockers = self._column_blockers(state)
        self._column_path_blockers = {}
        self._columns_needing_clear = set()
        max_segments = max((len(c.segments) for c in state.centipedes), default=1)
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        critical_row = max(0, safe_row_start - self.config.critical_head_rows)

        for centipede in state.centipedes:
            head = centipede.head
            cent_length = len(centipede.segments)
            size_bonus = self._small_centipede_focus(cent_length, max_segments)
            for segment in centipede.segments:
                seg_x, seg_y = segment
                value = (self.map_height - seg_y) * self.config.height_weight
                if segment == centipede.head:
                    value += self.config.head_bonus
                    value += size_bonus
                value -= abs(player_x - seg_x) * 0.5
                value -= column_blockers.get(seg_x, 0.0) * (
                    self.config.mushroom_penalty * self.config.column_blocker_penalty
                )
                blockers_on_path = self._path_blocker_count(
                    seg_x, state.player_pos[1], state.mushrooms
                )
                self._column_path_blockers[seg_x] = blockers_on_path
                if (
                    segment == centipede.head
                    and blockers_on_path >= self.config.stuck_blocker_threshold
                    and self._head_near_player(centipede.head, state.player_pos)
                ):
                    self._columns_needing_clear.add(seg_x)
                value -= blockers_on_path * (self.config.mushroom_penalty * 1.1)
                value -= threat_penalty(seg_x)
                column_scores[seg_x] = column_scores.get(seg_x, 0.0) + value

            dir_dx, _ = DIRECTION_TO_DELTA.get(centipede.direction, (0, 0))
            if dir_dx != 0:
                for step in range(1, self.config.head_prediction_horizon + 1):
                    predicted_x = head[0] + dir_dx * step
                    if not 0 <= predicted_x < self.map_width:
                        break
                    decay = 1 - (step / (self.config.head_prediction_horizon + 1))
                    bonus = self.config.head_bonus * 0.6 * decay
                    blockers_on_path = self._path_blocker_count(
                        predicted_x, state.player_pos[1], state.mushrooms
                    )
                    self._column_path_blockers.setdefault(predicted_x, blockers_on_path)
                    penalty = column_blockers.get(predicted_x, 0.0) * (
                        self.config.mushroom_penalty * self.config.column_blocker_penalty
                    )
                    penalty += blockers_on_path * (self.config.mushroom_penalty * 1.1)
                    penalty += threat_penalty(predicted_x)
                    column_scores[predicted_x] = (
                        column_scores.get(predicted_x, 0.0) + bonus - penalty
                    )

            for blocker in self._blocking_mushrooms(centipede, state.mushrooms):
                column_scores[blocker[0]] = (
                    column_scores.get(blocker[0], 0.0) + self.config.blocking_mushroom_bonus
                )
                self._columns_needing_clear.add(blocker[0])

            if head[1] >= critical_row:
                column_scores[head[0]] = (
                    column_scores.get(head[0], 0.0) + self.config.critical_head_bonus
                )

        self._column_path_blockers.setdefault(
            player_x, self._path_blocker_count(player_x, state.player_pos[1], state.mushrooms)
        )

        for column in self._trapped_columns.keys():
                column_scores[column] = column_scores.get(column, 0.0) + self.config.trapped_head_bonus
                self._columns_needing_clear.add(column)
                self._column_path_blockers.setdefault(
                    column,
                    self._path_blocker_count(column, state.player_pos[1], state.mushrooms),
                )

        safe_band_counts = self._safe_band_mushroom_counts(state)
        for column, count in safe_band_counts.items():
            if count <= 0:
                continue
            bonus = count * self.config.safe_band_clear_bonus
            column_scores[column] = column_scores.get(column, 0.0) + bonus
            if count >= self.config.safe_band_clear_threshold:
                self._columns_needing_clear.add(column)

        if not column_scores:
            return [player_x]

        sorted_columns = sorted(
            column_scores.items(), key=lambda item: item[1], reverse=True
        )
        best_score = sorted_columns[0][1]
        threshold = best_score * 0.6

        trapped_priority = sorted(
            self._trapped_columns.keys(), key=lambda col: abs(player_x - col)
        )

        targets: List[int] = []
        for column in trapped_priority:
            if column not in targets:
                targets.append(column)

        head = self._nearest_head(state)
        if head is not None:
            if head not in targets:
                targets.append(head)

        for column, score in sorted_columns:
            if score < threshold and column not in self._trapped_columns:
                continue
            if column not in targets:
                targets.append(column)
            if len(targets) >= 3:
                break

        if not targets:
            targets.append(player_x)

        filtered_targets = [
            col
            for col in targets
            if self._column_path_blockers.get(col, 0) <= 1
            or self._column_needs_clearing(col)
            or col in self._trapped_columns
        ]
        result = filtered_targets or targets
        self._target_columns_cache = list(result)
        self._target_columns_frame = self._frame
        return result

    def _nearest_head(self, state: GameSnapshot) -> Optional[int]:
        head_pos = self._nearest_head_position(state)
        return head_pos[0] if head_pos else None

    def _nearest_head_position(
        self, state: GameSnapshot
    ) -> Optional[Tuple[int, int]]:
        best_pos: Optional[Tuple[int, int]] = None
        best_distance = float("inf")
        for centipede in state.centipedes:
            head = centipede.head
            distance = abs(state.player_pos[0] - head[0]) + abs(
                state.player_pos[1] - head[1]
            )
            if distance < best_distance:
                best_distance = distance
                best_pos = head
        return best_pos

    def _head_near_player(
        self, head: Tuple[int, int], player_pos: Tuple[int, int]
    ) -> bool:
        horizontal = abs(head[0] - player_pos[0])
        vertical = abs(head[1] - player_pos[1])
        return vertical <= self.config.stuck_head_distance_rows or horizontal <= 2

    def _column_blockers(self, state: GameSnapshot) -> Dict[int, float]:
        blockers: Dict[int, float] = {}
        for (x, y), health in state.mushrooms.items():
            depth_factor = (self.map_height - y) / max(1, self.map_height)
            weight = 1.0 + (4 - health) * 0.25 + depth_factor * 0.5
            blockers[x] = blockers.get(x, 0.0) + weight
        return blockers

    def _path_blocker_count(
        self, column: int, player_y: int, mushrooms: Dict[Tuple[int, int], int]
    ) -> int:
        blockers = 0
        for y in range(player_y - 1, -1, -1):
            if (column, y) in mushrooms:
                blockers += 1
        return blockers

    def _should_descend(self, state: GameSnapshot) -> bool:
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        if state.player_pos[1] >= safe_row_start:
            return False

        column = state.player_pos[0]
        blockers_in_column = sum(
            1
            for (mx, my) in state.mushrooms.keys()
            if mx == column and my >= state.player_pos[1] - 1
        )
        if blockers_in_column >= self.config.stuck_blocker_threshold:
            aligned_heads = [
                cent.head
                for cent in state.centipedes
                if cent.head[0] == column and cent.head[1] <= state.player_pos[1]
            ]
            if aligned_heads:
                return False
        return True

    def _alignment_row(self) -> int:
        target = self.map_height - 1 - self.config.alignment_row_offset
        return max(0, target)

    def _preferred_row(self, state: GameSnapshot, any_low: bool) -> int:
        if any_low:
            return max(0, self.map_height - self.config.target_band_rows - 3)
        return self._alignment_row()

    def _small_centipede_focus(self, length: int, max_length: int) -> float:
        if max_length <= 0:
            return 0.0
        advantage = max(0, max_length - length) / max_length
        return advantage * self.config.small_centipede_bonus

    def _blocking_mushrooms(
        self, cent: Optional[CentipedeSnapshot], mushrooms: Dict[Tuple[int, int], int]
    ) -> List[Tuple[int, int]]:
        if not cent:
            return []
        head_x, head_y = cent.head
        candidates = [
            (head_x - 1, head_y),
            (head_x + 1, head_y),
            (head_x, head_y + 1),
            (head_x, head_y + 2),
        ]
        blockers: List[Tuple[int, int]] = []
        for pos in candidates:
            if self._in_bounds(pos) and pos in mushrooms:
                blockers.append(pos)
        return blockers

    def _blocking_mushroom_target(
        self, cent: Optional[CentipedeSnapshot], state: GameSnapshot
    ) -> Optional[ShotCandidate]:
        blockers = self._blocking_mushrooms(cent, state.mushrooms)
        if not cent or not blockers:
            return None
        px, py = state.player_pos
        best: Optional[Tuple[int, int]] = None
        best_distance = float("inf")
        for bx, by in blockers:
            if bx != px:
                continue
            if by >= py:
                continue
            if not self._path_clear_for_shot(bx, py, by, state.mushrooms):
                continue
            distance = py - by
            if distance < best_distance:
                best_distance = distance
                best = (bx, by)
        if not best:
            return None
        score = (
            self.config.mushroom_clear_bonus
            + self.config.stuck_clear_bonus
            + self.config.blocking_mushroom_bonus
        )
        frames_until_hit = state.player_pos[1] - best[1]
        return ShotCandidate(
            target=best,
            score=score,
            type="mushroom",
            is_head=False,
            frames_until_hit=frames_until_hit,
        )

    def _column_needs_clearing(self, column: int) -> bool:
        return column in self._columns_needing_clear

    def _preemptive_column_clear_target(
        self, state: GameSnapshot
    ) -> Optional[ShotCandidate]:
        px, py = state.player_pos
        max_rows = min(py, self.config.preemptive_column_clear_rows)
        blockers: List[Tuple[int, int]] = []
        for offset in range(1, max_rows + 1):
            pos = (px, py - offset)
            if pos in state.mushrooms:
                blockers.append(pos)
        if len(blockers) < self.config.preemptive_column_clear_threshold:
            return None
        target = blockers[0]
        frames = state.player_pos[1] - target[1]
        score = (
            self.config.safe_band_clear_bonus
            + len(blockers) * 30.0
            + (self.config.preemptive_column_clear_rows - len(blockers)) * 10.0
        )
        return ShotCandidate(
            target=target,
            score=score,
            type="mushroom",
            is_head=False,
            frames_until_hit=frames,
        )

    def _parse_state(self, state: Dict) -> Optional[GameSnapshot]:
        bug = state.get("bug_blaster")
        if not bug or "pos" not in bug:
            return None

        mushrooms: Dict[Tuple[int, int], int] = {}
        centipedes: List[CentipedeSnapshot] = []
        centipede_tiles: Set[Tuple[int, int]] = set()
        segment_lookup: Dict[Tuple[int, int], CentipedeSnapshot] = {}

        for mushroom in state.get("mushrooms", []):
            if "pos" not in mushroom:
                continue
            pos = tuple(mushroom["pos"])
            mushrooms[pos] = mushroom.get("health", 4)

        for cent_data in state.get("centipedes", []):
            body = [tuple(seg) for seg in cent_data.get("body", [])]
            if not body:
                continue

            cent_snapshot = CentipedeSnapshot(
                name=cent_data.get("name", "centipede"),
                body=body,
                direction=cent_data.get("direction", 1),
            )
            centipedes.append(cent_snapshot)
            centipede_tiles.update(body)
            for segment in body:
                segment_lookup[segment] = cent_snapshot

        return GameSnapshot(
            player_pos=tuple(bug["pos"]),
            player_alive=bug.get("alive", True),
            mushrooms=mushrooms,
            centipedes=centipedes,
            centipede_tiles=centipede_tiles,
            segment_lookup=segment_lookup,
        )

    def _compute_threat_tiles(
        self, state: GameSnapshot
    ) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
        fatal = set(state.centipede_tiles)
        short_term = set(fatal)
        medium_term: Set[Tuple[int, int]] = set()
        for seg_x, seg_y in fatal:
            for delta in MOVE_TO_DELTA.values():
                neighbor = (seg_x + delta[0], seg_y + delta[1])
                if self._in_bounds(neighbor):
                    short_term.add(neighbor)

        horizon = max(self.config.medium_term_horizon, self.config.prediction_steps)
        for cent in state.centipedes:
            future_positions = self._simulate_centipede_future_positions(
                cent, state.mushrooms, horizon
            )
            for step, pos in enumerate(future_positions, start=1):
                if self._in_bounds(pos):
                    if step <= self.config.short_term_horizon:
                        short_term.add(pos)
                    elif step <= self.config.medium_term_horizon:
                        medium_term.add(pos)

        return fatal, short_term, medium_term

    def _incoming_column_threats(self, state: GameSnapshot) -> Dict[int, int]:
        threats: Dict[int, int] = {}
        px, py = state.player_pos
        horizon = max(1, self.config.column_threat_horizon)
        for cent in state.centipedes:
            predicted = self._simulate_centipede_future_positions(
                cent, state.mushrooms, horizon
            )
            for step, (cx, cy) in enumerate(predicted, start=1):
                if cy > py:
                    continue
                if (py - cy) > self.config.near_head_clear_distance:
                    continue
                current = threats.get(cx)
                if current is None or step < current:
                    threats[cx] = step
                break
        return threats

    def _detect_trapped_columns(
        self, state: GameSnapshot
    ) -> Dict[int, Tuple[CentipedeSnapshot, Tuple[Tuple[int, int], ...]]]:
        trapped: Dict[int, Tuple[CentipedeSnapshot, Tuple[Tuple[int, int], ...]]] = {}
        for cent in state.centipedes:
            hx, hy = cent.head
            dx, dy = DIRECTION_TO_DELTA.get(cent.direction, (0, 0))
            forward = (hx + dx, hy + dy)
            forward_blocked = (
                not self._in_bounds(forward)
                or forward in state.mushrooms
                or forward in state.centipede_tiles
            )
            vertical_blockers: List[Tuple[int, int]] = []
            for offset in (1, 2):
                pos = (hx, hy + offset)
                if not self._in_bounds(pos):
                    break
                if pos in state.mushrooms:
                    vertical_blockers.append(pos)
                else:
                    break
            if not vertical_blockers:
                continue
            if dy == 0 and not forward_blocked:
                continue
            trapped[hx] = (cent, tuple(vertical_blockers))
        return trapped

    def _safe_band_mushroom_counts(self, state: GameSnapshot) -> Dict[int, int]:
        safe_row_start = max(0, self.map_height - self.config.target_band_rows)
        counts: Dict[int, int] = {}
        for (mx, my) in state.mushrooms.keys():
            if my >= safe_row_start:
                counts[mx] = counts.get(mx, 0) + 1
        return counts

    def _low_space(self, state: GameSnapshot) -> bool:
        px, py = state.player_pos
        safe_moves = 0
        for delta in MOVE_TO_DELTA.values():
            nxt = (px + delta[0], py + delta[1])
            if self._is_tile_safe(nxt, state):
                safe_moves += 1
        threshold = max(0, self.config.low_space_threshold)
        return safe_moves < threshold

    def _simulate_centipede_future_states(
        self,
        cent: CentipedeSnapshot,
        mushrooms: Dict[Tuple[int, int], int],
        steps: int,
    ) -> List[List[Tuple[int, int]]]:
        if steps <= 0 or not cent.body:
            return []
        steps = min(steps, self._simulation_cache_steps)
        cache = self._get_centipede_state_sequence(cent, mushrooms)
        if not cache:
            return []
        return cache[:steps]

    def _simulate_centipede_future_positions(
        self,
        cent: CentipedeSnapshot,
        mushrooms: Dict[Tuple[int, int], int],
        steps: int,
    ) -> List[Tuple[int, int]]:
        if steps <= 0 or not cent.body:
            return []
        steps = min(steps, self._simulation_cache_steps)
        cent_key = id(cent)
        cache = self._get_centipede_state_sequence(cent, mushrooms)
        if not cache:
            return []
        if cent_key not in self._centipede_head_cache:
            self._centipede_head_cache[cent_key] = [segments[-1] for segments in cache]
        heads = self._centipede_head_cache[cent_key]
        return heads[:steps]

    def _get_centipede_state_sequence(
        self, cent: CentipedeSnapshot, mushrooms: Dict[Tuple[int, int], int]
    ) -> List[List[Tuple[int, int]]]:
        if self._simulation_cache_frame != self._frame:
            self._prepare_frame_caches()
        cent_key = id(cent)
        if cent_key not in self._centipede_state_cache:
            self._centipede_state_cache[cent_key] = self._generate_centipede_states(
                cent, mushrooms, self._simulation_cache_steps
            )
        return self._centipede_state_cache.get(cent_key, [])

    def _generate_centipede_states(
        self,
        cent: CentipedeSnapshot,
        mushrooms: Dict[Tuple[int, int], int],
        steps: int,
    ) -> List[List[Tuple[int, int]]]:
        if steps <= 0 or not cent.body:
            return []

        states: List[List[Tuple[int, int]]] = []
        segments: List[Tuple[int, int]] = list(cent.body)
        mush_pos = set(mushrooms.keys())
        dir_local = cent.direction
        move_dir = 1
        head = segments[-1]
        if head[1] == 0:
            move_dir = 1
        elif head[1] >= (self.map_height - 1):
            move_dir = -1
        if len(segments) >= 2:
            sec = segments[-2]
            if sec[1] > head[1]:
                move_dir = -1
            elif sec[1] < head[1]:
                move_dir = 1

        waiting_to_move_vertically = False
        for _ in range(steps):
            head_x, head_y = segments[-1]
            dx, dy = DIRECTION_TO_DELTA.get(dir_local, (0, 0))
            attempted = (head_x + dx, head_y + dy)
            horizontal_blocked = (
                not self._in_bounds(attempted) or attempted in mush_pos
            )
            if horizontal_blocked or waiting_to_move_vertically:
                vertical_target = (head_x, head_y + move_dir)
                if self._in_bounds(vertical_target) and vertical_target not in mush_pos:
                    new_head = vertical_target
                    waiting_to_move_vertically = False
                else:
                    new_head = (head_x, head_y)
                    waiting_to_move_vertically = True
                dir_local = self._invert_direction(dir_local)
            else:
                new_head = attempted
                waiting_to_move_vertically = False

            segments = segments[1:] + [new_head]
            states.append(list(segments))
            move_dir = self._update_vertical_travel_direction(segments, move_dir)

        return states

    def _path_clear_for_shot(self, column: int, from_y: int, to_y: int, mushrooms: Dict[Tuple[int, int], int]) -> bool:
        if to_y >= from_y:
            return False
        for y in range(to_y + 1, from_y):
            if (column, y) in mushrooms:
                return False
        return True

    def _has_intercept_window(self, column: int, from_y: int, state: GameSnapshot) -> bool:
        return self._find_intercept_match(column, from_y, state) is not None

    def _can_hit_now(self, state: GameSnapshot) -> bool:
        px, py = state.player_pos
        return self._has_intercept_window(px, py, state)

    def _find_intercept_match(
        self,
        column: int,
        from_y: int,
        state: GameSnapshot,
        arrival_delay: int = 0,
        target_centipede: Optional[CentipedeSnapshot] = None,
        max_frames: Optional[int] = None,
    ) -> Optional[Tuple[int, Tuple[int, int], CentipedeSnapshot]]:
        if from_y <= 0:
            return None
        tolerance = max(0, self.config.intercept_tolerance_frames)
        centipedes = [target_centipede] if target_centipede else state.centipedes
        if not centipedes:
            return None
        base_horizon = (
            max_frames if max_frames is not None else self.config.prediction_steps
        )
        base_horizon = max(0, base_horizon)
        horizon = min(self.config.max_path_length, arrival_delay + base_horizon)
        if horizon <= 0:
            return None

        best_match: Optional[Tuple[int, Tuple[int, int], CentipedeSnapshot]] = None
        for cent in centipedes:
            states = self._simulate_centipede_future_states(
                cent, state.mushrooms, horizon
            )
            match_found = False
            for absolute_t, segments in enumerate(states, start=1):
                if absolute_t <= arrival_delay:
                    continue
                for seg_x, seg_y in segments:
                    if seg_x != column or seg_y >= from_y:
                        continue
                    bullet_travel = from_y - seg_y
                    dt = absolute_t - arrival_delay
                    if bullet_travel <= 0:
                        continue
                    if abs(bullet_travel - dt) <= tolerance:
                        if self._path_clear_for_shot(
                            column, from_y, seg_y, state.mushrooms
                        ):
                            match_frames = dt
                            if (
                                best_match is None
                                or match_frames < best_match[0]
                            ):
                                best_match = (match_frames, (seg_x, seg_y), cent)
                            match_found = True
                            break
                if match_found:
                    break
            if best_match and best_match[0] <= 1:
                break

        return best_match

    def _intercept_window_exists(
        self,
        column: int,
        from_y: int,
        state: GameSnapshot,
        arrival_delay: int = 0,
        target_centipede: Optional[CentipedeSnapshot] = None,
        max_frames: Optional[int] = None,
    ) -> bool:
        return (
            self._find_intercept_match(
                column,
                from_y,
                state,
                arrival_delay=arrival_delay,
                target_centipede=target_centipede,
                max_frames=max_frames,
            )
            is not None
        )

    def _find_intercept_plan(self, state: GameSnapshot) -> Optional[List[str]]:
        start = state.player_pos
        blocked = set(state.mushrooms) | set(state.centipede_tiles)
        queue: Deque[Tuple[Tuple[int, int], List[str]]] = deque([(start, [])])
        visited: Set[Tuple[int, int]] = {start}
        while queue:
            pos, path = queue.popleft()
            if len(path) > self.config.max_path_length:
                continue
            arrival = len(path)
            if self._intercept_window_exists(pos[0], pos[1], state, arrival_delay=arrival):
                return path
            for move, delta in MOVE_TO_DELTA.items():
                nxt = (pos[0] + delta[0], pos[1] + delta[1])
                if nxt in visited or nxt in blocked:
                    continue
                if not self._in_bounds(nxt):
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [move]))
        return None

    def _try_aggressive_ascent(self, state: GameSnapshot) -> Optional[List[str]]:
        px, py = state.player_pos
        current_shot_score = self._simulated_best_shot_score(state)
        best_plan: Optional[List[str]] = None
        best_gain = 0.0
        head_pos = self._nearest_head_position(state)
        head_y = head_pos[1] if head_pos else None
        max_travel = min(self.config.max_vertical_travel_up, py)
        for step in range(1, max_travel + 1):
            path_safe = True
            target_y = py - step
            if head_y is not None and target_y <= head_y:
                break
            for s in range(1, step + 1):
                pos = (px, py - s)
                if (
                    pos in state.mushrooms
                    or pos in self._danger_tiles
                    or pos in self._medium_danger_tiles
                ):
                    path_safe = False
                    break
            if not path_safe:
                break
            sim_state = GameSnapshot(
                player_pos=(px, py - step),
                player_alive=state.player_alive,
                mushrooms=state.mushrooms,
                centipedes=state.centipedes,
                centipede_tiles=state.centipede_tiles,
                segment_lookup=state.segment_lookup,
            )
            sim_score = self._simulated_best_shot_score(sim_state)
            gain = sim_score - current_shot_score
            if gain >= self.config.aggressive_min_score_gain:
                return ["w"] * step
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_plan = ["w"] * step
        return best_plan

    def _simulated_best_shot_score(self, state: GameSnapshot) -> float:
        can_hit = self._has_intercept_window(state.player_pos[0], state.player_pos[1], state)
        candidate = self._select_shot_candidate(state, can_hit_now=can_hit)
        return candidate.score if candidate else 0.0

    def _invert_direction(self, direction: int) -> int:
        if direction == 1:
            return 3
        if direction == 3:
            return 1
        if direction == 0:
            return 2
        if direction == 2:
            return 0
        return direction

    def _update_vertical_travel_direction(
        self, segments: Sequence[Tuple[int, int]], current_dir: int
    ) -> int:
        if not segments:
            return current_dir or 1
        head_y = segments[-1][1]
        if head_y <= 0:
            return 1
        if head_y >= self.map_height - 1:
            return -1
        if len(segments) >= 2:
            prev_y = segments[-2][1]
            if prev_y > head_y:
                return -1
            if prev_y < head_y:
                return 1
        return current_dir or 1

    def _shot_ready(self) -> bool:
        return (self._frame - self._last_shot_frame) >= self.config.shot_cooldown_frames

    def _record_shot(self) -> None:
        self._last_shot_frame = self._frame

    def _apply_move(self, pos: Tuple[int, int], move: str) -> Tuple[int, int]:
        delta = MOVE_TO_DELTA.get(move, (0, 0))
        return pos[0] + delta[0], pos[1] + delta[1]

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.map_width and 0 <= pos[1] < self.map_height

    def _is_tile_safe(
        self, pos: Tuple[int, int], state: GameSnapshot, allow_medium: bool = False
    ) -> bool:
        if not self._in_bounds(pos):
            return False
        if pos in state.mushrooms or pos in self._fatal_tiles:
            return False
        if pos in self._danger_tiles:
            return False
        if not allow_medium and pos in self._medium_danger_tiles:
            return False
        return True

    @staticmethod
    def _min_distance(
        pos: Tuple[int, int], centipede_tiles: Iterable[Tuple[int, int]]
    ) -> float:
        if not centipede_tiles:
            return 10.0
        distances = [abs(pos[0] - x) + abs(pos[1] - y) for x, y in centipede_tiles]
        return float(min(distances))


async def agent_loop(
    server_address: str = "localhost:8000", agent_name: str = "student"
) -> None:
    agent = StudentAgent()
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


# DO NOT CHANGE THE LINES BELLOW
# You can change the default values using the command line, example:
# $ NAME='arrumador' python3 client.py
loop = asyncio.get_event_loop()
SERVER = os.environ.get("SERVER", "localhost")
PORT = os.environ.get("PORT", "8000")
NAME = os.environ.get("NAME", getpass.getuser())
loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))
