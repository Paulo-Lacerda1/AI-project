import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from consts import Direction
from game import Game, Mushroom, Centipede


@pytest.mark.asyncio
async def test_centipede_moves_down_after_trap():
    """Ensure centipede is not stuck when blocked by mushrooms while moving down."""

    game = Game(timeout=300)
    game.start(["tester"])

    game._mushrooms = []
    game._centipedes = []

    cx = game.map.size[0] // 2
    cy = game.map.size[1] // 2
    centipede_body = [
        (cx, cy),
        (cx - 1, cy),
        (cx - 2, cy),
        (cx - 3, cy),
    ]
    centipede = Centipede("debug_down", centipede_body, Direction.EAST)
    game._centipedes.append(centipede)

    game._mushrooms = [
        Mushroom(cx + 6, cy),
        Mushroom(cx + 5, cy + 1),
        Mushroom(cx - 9, cy),
        Mushroom(cx - 8, cy + 1),
    ]

    initial_y = centipede.head[1]

    for _ in range(60):
        await game.next_frame()

    final_y = centipede.head[1]

    assert final_y != initial_y, (
        f"Centipede stuck at Y={initial_y}, did not move vertically"
    )


@pytest.mark.asyncio
async def test_centipede_moves_up_after_trap():
    """Ensure centipede is not stuck when blocked by mushrooms while moving up."""

    game = Game(timeout=300)
    game.start(["tester"])

    game._mushrooms = []
    game._centipedes = []

    cx = game.map.size[0] // 2
    cy = game.map.size[1] // 2
    centipede_body = [
        (cx, cy),
        (cx - 1, cy),
        (cx - 2, cy),
        (cx - 3, cy),
    ]
    centipede = Centipede("debug_down", centipede_body, Direction.EAST)
    centipede.move_dir = -1  # moving up
    game._centipedes.append(centipede)

    game._mushrooms = [
        Mushroom(cx + 6, cy),
        Mushroom(cx + 5, cy - 1),
        Mushroom(cx - 9, cy),
        Mushroom(cx - 8, cy - 1),
    ]

    initial_y = centipede.head[1]

    for _ in range(60):
        await game.next_frame()

    final_y = centipede.head[1]

    assert final_y != initial_y, (
        f"Centipede stuck at Y={initial_y}, did not move vertically"
    )
