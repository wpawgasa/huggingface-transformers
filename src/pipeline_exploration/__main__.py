"""Allow running as: python -m src.pipeline_exploration [args]."""

import sys

from src.pipeline_exploration.experiment_runner import main

sys.exit(main())
