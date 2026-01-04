from __future__ import annotations
import asyncio, random
from academy.agent import Agent, action, loop
from academy.handle import Handle
from academy.identifier import AgentId
from player import BattleshipPlayer

class TournamentAgent(Agent):
    def __init__(self):
        super().__init__()
        self.players: dict[AgentId[BattleshipPlayer], Handle[BattleshipPlayer]] = {}
        self.elo_scores: dict[AgentId[BattleshipPlayer], float] = {}
        self.k_factor = 32

    @action
    async def register(self, player: Handle[BattleshipPlayer]) -> None:
        player_id = player.id
        self.players[player_id] = player
        self.elo_scores[player_id] = 1000.0
        print(f"[Tournament] Registered player {player_id} with ELO 1000")

    @loop
    async def run(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            if len(self.players) < 2:
                await asyncio.sleep(1)
                continue

            player_ids = list(self.players.keys())
            player_0_id, player_1_id = random.sample(player_ids, 2)
            player_0, player_1 = self.players[player_0_id], self.players[player_1_id]

            print(f"[Tournament] Match: {player_0_id} vs {player_1_id}")

            # ---- COPY GAME LOGIC FROM Coordinator.py ----
            winner = await self._run_game(player_0, player_1)

            # Update ELO
            self._update_elo(player_0_id, player_1_id, winner)

            await asyncio.sleep(2)  # Prevents busy loop

    async def _run_game(self, player_0, player_1) -> int:
        """Replace this with actual Coordinator.py game logic."""
        # Example placeholder:
        winner = random.choice([0, 1])
        return winner

    def _update_elo(self, id0: AgentId, id1: AgentId, winner: int):
        r0, r1 = self.elo_scores[id0], self.elo_scores[id1]
        e0 = 1 / (1 + 10 ** ((r1 - r0) / 400))
        e1 = 1 / (1 + 10 ** ((r0 - r1) / 400))

        if winner == 0:
            s0, s1 = 1, 0
        else:
            s0, s1 = 0, 1

        self.elo_scores[id0] = r0 + self.k_factor * (s0 - e0)
        self.elo_scores[id1] = r1 + self.k_factor * (s1 - e1)

        print(f"[Tournament] Updated ELOs: {id0}={self.elo_scores[id0]:.1f}, {id1}={self.elo_scores[id1]:.1f}")

    @action
    async def report(self) -> dict[AgentId[BattleshipPlayer], float]:
        return self.elo_scores
