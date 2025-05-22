"""
Módulo de consolidação de dados do experimento de noisy neighbors.

Este módulo fornece funções para carregar e consolidar dados de métricas
de diferentes tenants, fases e rounds do experimento.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import re

# As funções list_available_tenants, list_available_metrics, parse_timestamp, 
# load_metric_data, load_multiple_metrics, load_experiment_data, e select_tenants 
# foram movidas para refactor/data_handling/loader.py
