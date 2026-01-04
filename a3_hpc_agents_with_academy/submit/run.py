#%%
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from academy.logging import init_logging
from academy.manager import Manager
from academy.exchange import LocalExchangeFactory

from player import BattleshipPlayer # player.py is with the run.py file
from coordinator import Coordinator

#%%
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
        print(player_1, player_2)
        coordinator = await manager.launch(
            Coordinator,
            args=(player_1, player_2),
        )

        loop = asyncio.get_event_loop()
        while True:
            user_input = await loop.run_in_executor(
                None,
                input,
                'Enter command (exit, game, stat): ',
            )
            if user_input.lower() == 'exit':
                print('Exiting...')
                break
            elif user_input.lower() == 'game':
                game = await coordinator.get_game_state()
                print('Current Game State: ')
                print(game)
            elif user_input.lower() == 'stat':
                stats = await coordinator.get_player_stats()
                print(f'Player 0 has won {stats[0]} games')
                print(f'Player 1 has won {stats[1]} games')
            else:
                print('Unknown command')
            print('-----------------------------------------------------')

        # Upon exit, the Manager context will instruct each agent to shutdown,
        # closing their respective handles, and shutting down the executors.

    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
# %%
