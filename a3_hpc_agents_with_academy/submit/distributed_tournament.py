from __future__ import annotations
import asyncio
import logging
import uuid

from academy.logging import init_logging
from academy.manager import Manager
from academy.exchange import HttpExchangeFactory
from globus_compute_sdk import Executor as GCExecutor

from player import BattleshipPlayer
from simple_tournament import TournamentAgent

logger = logging.getLogger(__name__)

ENDPOINT_ID = "8c638284-e085-4744-88b4-952162ac6d7d"

EXCHANGE_URL = "https://exchange.academy-agents.org"

async def main() -> int:
    init_logging(logging.INFO)
    
    executor = GCExecutor(endpoint_id=ENDPOINT_ID)

    factory = HttpExchangeFactory(url=EXCHANGE_URL, auth_method='globus')

    async with await Manager.from_exchange_factory(factory=factory, executors=executor) as manager:
        # Launch two players and a tournament agent
        player_1 = await manager.launch(BattleshipPlayer)
        player_2 = await manager.launch(BattleshipPlayer)
        tournament = await manager.launch(TournamentAgent)

        await tournament.register(player_1)
        await tournament.register(player_2)

        loop = asyncio.get_event_loop()
        while True:
            user_input = await loop.run_in_executor(None, input, 'Enter command (exit, game, stat): ')
            if user_input.lower() == 'exit':
                print('Exiting...')
                break
            elif user_input.lower() == 'game':
                report = await tournament.report()
                print('Current ELO rankings:')
                print(report)
            else:
                print('Unknown command')
            print('------------------------------------------')

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

