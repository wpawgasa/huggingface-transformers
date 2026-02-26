"""Allow running as: python -m src.architecture_deepdive [args]."""

import sys

from src.architecture_deepdive.experiment_runner import main

sys.exit(main())
