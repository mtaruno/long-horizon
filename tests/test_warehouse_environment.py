import matplotlib

matplotlib.use("Agg")  # headless backend for tests

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.warehouse import WarehouseEnv
from src.utils.visualization import create_evaluation_animation



# load config
with open("config/warehouse_v1.yaml", 'r') as f:
    config = yaml.safe_load(f)

# create path object and goal


env=WarehouseEnv(config)

create_evaluation_animation(env, path, goal, filename="visualizations/03_final_run.gif")