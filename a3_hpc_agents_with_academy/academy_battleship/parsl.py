from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider
from parsl.usage_tracking.levels import LEVEL_1

def get_htex_local_config():
    local_config = Config(
        executors=[
            HighThroughputExecutor(
                label='Local_HTEX',
                max_workers_per_node=8
            )
        ],
        usage_tracking=LEVEL_1,
    )

    return local_config

def get_htex_midway_config():
    midway_config = Config(
        executors=[
            HighThroughputExecutor(
                label='Midway_HTEX_multinode',
                address=address_by_interface('bond0'),
                worker_debug=False,
                max_workers_per_node=48,
                provider=SlurmProvider(
                    'caslake',
                    launcher=SrunLauncher(),
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    scheduler_options='',
                    # Command to be run before starting a worker, such as:
                    # 'module load Anaconda; source activate parsl_env'.
                    worker_init='', # TODO: Fill in command to activate venv
                    walltime='00:30:00'
                ),
            )
        ],
        usage_tracking=LEVEL_1,
    )

    return midway_config

PARSL_CONFIGS = {
    'htex-local': get_htex_local_config,
    'htex-midway': get_htex_midway_config,
}