from __future__ import annotations
import asyncio

from academy.agent import Agent
from academy.agent import action
from academy.agent import loop
from academy.handle import Handle
from academy.indentifier import AgentId

from academy_battleship.player import BattleshipPlayer

class TournamentAgent(Agent):
    def __init__(self):
        super().__init__()
        ...
        
    @action
    async def register(self, player: Handle[BattleshipPlayer]) -> None:
        ...

    @loop
    async def loop(self, shutdown: asyncio.Event):
        ...

    @action
    async def report(self) -> dict[AgentId[BattleshipPlayer], float]:
        ...
