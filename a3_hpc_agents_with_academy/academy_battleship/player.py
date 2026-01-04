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
    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.guesses = Board()

    @action
    async def get_move(self) -> Crd:
        await asyncio.sleep(1)
        return Crd(0, 0)

    @action
    async def notify_result(
        self,
        loc: Crd,
        result: Literal['hit', 'miss', 'guessed'],
    ):
        ""

    @action
    async def notify_move(self, loc: Crd) -> None:
        ...

    @action
    async def new_game(self, ships: list[int], size: int = 10) -> Board:
        return Board()
# %%
