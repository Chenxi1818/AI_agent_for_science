#%%
from __future__ import annotations

import asyncio
import logging
import random
from typing import Literal

from academy.agent import action
from academy.agent import Agent

from battleship import Board
from battleship import Crd

#%%
logger = logging.getLogger(__name__)


class BattleshipPlayer(Agent):
    """A simple Battleship player agent that plays randomly but avoids repeating guesses."""
    
    def __init__(self) -> None:
        super().__init__()
        self.size = 10
        self.available_moves: set[Crd] = set()
        self.previous_guesses: list[Crd] = []
        self.last_guess: Crd | None = None
        self.own_board: Board | None = None

    @action
    async def new_game(self, ships: list[int], size: int = 10) -> Board:
        """
        Called when a new game starts. The player should return a new Board
        with all ships placed randomly.
        """
        logger.info("Starting new game with board size %dx%d", size, size)
        self.size = size
        self.previous_guesses.clear()
        self.available_moves = {Crd(x, y) for x in range(size) for y in range(size)}

        board = Board(size=size)

        for ship_len in ships:
            placed = False
            while not placed:
                start = Crd(random.randrange(size), random.randrange(size))
                direction = random.choice(['horizontal', 'vertical'])
                placed = board.place_ship(start, ship_len, direction) is not None

        self.own_board = board
        logger.info("Ships placed successfully: %d ships", len(board.ships))
        return board
    
    @action
    async def get_move(self) -> Crd:
        """
        Return a (x, y) coordinate guess where an opposing ship might be.
        Ensures no duplicates and random guessing strategy.
        """
        await asyncio.sleep(1)  # simulate some "thinking time"

        if not self.available_moves:
            raise RuntimeError("No available moves left!")

        move = random.choice(list(self.available_moves))
        self.available_moves.remove(move)
        self.previous_guesses.append(move)
        self.last_guess = move

        logger.debug("Player guesses %s", move)
        return move

    @action
    async def notify_result(
        self,
        loc: Crd,
        result: Literal['hit', 'miss', 'guessed'],
    ):
        """
        Notify the agent of the result of its last move.
        Can be extended for smarter strategies later.
        """
        logger.info("Result of our move at %s: %s", loc, result)

        # Optional logic for smarter strategies later:
        if result == "hit":
            # You could try adding nearby cells to a priority list
            logger.debug("Hit registered — might target nearby cells next.")
        elif result == "miss":
            logger.debug("Miss registered.")
        elif result == "guessed":
            logger.warning("Duplicate guess detected at %s", loc)

    @action
    async def notify_move(self, loc: Crd) -> None:
        """
        Notify the agent of an opponent's move.
        """
        logger.info("Opponent guessed %s", loc)
        if self.own_board:
            result = self.own_board.receive_attack(loc)
            logger.debug("Opponent move result: %s", result)

    
# %%
