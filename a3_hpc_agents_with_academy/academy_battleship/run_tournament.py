from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from academy.logging import init_logging
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

from player import BattleshipPlayer
from simple_tournamentv2 import TournamentAgent


logger = logging.getLogger(__name__)


async def main() -> int:
    init_logging(logging.INFO)

    executor = ThreadPoolExecutor(max_workers=3)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=executor,
    ) as manager:
        # Launch each of the three agents, each implementing a different
        # behavior. The returned type is a handle to that agent used to
        # invoke actions.
        player_1 = await manager.launch(BattleshipPlayer)
        player_2 = await manager.launch(BattleshipPlayer)
        tournament = await manager.launch(TournamentAgent)

        await tournament.register(player_1)
        await tournament.register(player_2)

        loop = asyncio.get_event_loop()
        while True:
            user_input = await loop.run_in_executor(
                None,
                input,
                'Enter command (exit, game): ',
            )
            if user_input.lower() == 'exit':
                print('Exiting...')
                break
            elif user_input.lower() == 'game':
                report = await tournament.report()
                print('Current ELO rankings: ')
                print(report)
            else:
                print('Unknown command')
            print('-----------------------------------------------------')

        # Upon exit, the Manager context will instruct each agent to shutdown,
        # closing their respective handles, and shutting down the executors.

    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))