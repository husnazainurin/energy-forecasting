# Import relevant packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Get parent address
PARENT_ADDRESS = Path(os.getcwd()).parent

# Import csv file
energy = pd.read_csv(f'{PARENT_ADDRESS}/data/raw/energydata_complete.csv')