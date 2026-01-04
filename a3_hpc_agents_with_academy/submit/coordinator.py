from __future__ import annotations

import asyncio
import logging
from typing import ClassVar

from academy.agent import action, Agent, loop
from academy.handle import Handle

from battleship import Game, Board
from player import BattleshipPlayer

logger = logging.getLogger(__name__)


class Coordinator(Agent):
    """Coordinator agent that manages repeated Battleship games between two player agents."""

    _default_ships: ClassVar[list[int]] = [5, 5, 4, 3, 2]

    def __init__(
        self,
        player_0: Handle[BattleshipPlayer],
        player_1: Handle[BattleshipPlayer],
        *,
        size: int = 10,
        ships: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.player_0 = player_0
        self.player_1 = player_1
        self.size = size
        self.ships = ships or self._default_ships
        self.game_state = Game(Board(size), Board(size))
        self.stats = [0, 0]

    async def game(self, shutdown: asyncio.Event) -> int:
        """Run one full game between player_0 and player_1."""
        while not shutdown.is_set():
            # --- Player 0's turn ---
            attack = await self.player_0.get_move()
            result = self.game_state.attack(0, attack)
            await self.player_0.notify_result(attack, result)
            await self.player_1.notify_move(attack)
            winner = self.game_state.check_winner()
            if winner >= 0:
                return winner

            # --- Player 1's turn ---
            attack = await self.player_1.get_move()
            result = self.game_state.attack(1, attack)
            await self.player_1.notify_result(attack, result)
            await self.player_0.notify_move(attack)
            winner = self.game_state.check_winner()
            if winner >= 0:
                return winner

        return -1  # stopped externally

    @loop
    async def play_games(self, shutdown: asyncio.Event) -> None:
        """Continuous loop that plays games until shutdown."""
        while not shutdown.is_set():
            # Get new boards from each player
            player_0_board = await self.player_0.new_game(self.ships, self.size)
            player_1_board = await self.player_1.new_game(self.ships, self.size)

            # Start a new game
            self.game_state = Game(player_0_board, player_1_board)
            logger.info("Starting new game!")

            winner = await self.game(shutdown)
            if winner >= 0:
                self.stats[winner] += 1
                logger.info("Game finished. Winner: Player %d", winner)
            else:
                logger.info("Game interrupted by shutdown.")

    @action
    async def get_game_state(self) -> Game | None:
        """Get the current Game state."""
        return self.game_state

    @action
    async def get_player_stats(self) -> list[int]:
        """Get total win counts for each player."""
        return self.stats
