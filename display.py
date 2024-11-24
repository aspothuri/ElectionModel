import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from model import predict

# Load the prediction data
predictions = predict()
print(predictions)
us_states = gpd.read_file('us_states.geojson')

us_states = us_states.rename(columns={'name': 'state'}) 
us_states = us_states.rename(columns={'NAME': 'state'})

us_states = us_states.merge(predictions, on='state', how='left')

def categorize_margin(margin):
    if margin > 10:
        return 'safe blue'
    elif margin > 5:
        return 'likely blue'
    elif margin > 1:
        return 'lean blue'
    elif margin < -10:
        return 'safe red'
    elif margin < -5:
        return 'likely red'
    elif margin < -1:
        return 'lean red'
    else:
        return 'tilt blue' if margin > 0 else 'tilt red'

# Calculate the margin and categorize each state
us_states['margin'] = us_states['prediction_2024']
us_states['category'] = us_states['margin'].apply(categorize_margin)

color_map = {
    'safe blue': 'darkblue',
    'likely blue': 'blue',
    'lean blue': 'lightblue',
    'tilt blue': 'skyblue',
    'safe red': 'darkred',
    'likely red': 'red',
    'lean red': 'lightcoral',
    'tilt red': 'salmon'
}

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
us_states.plot(column='category', cmap=None, color=us_states['category'].map(color_map), linewidth=0.8, ax=ax, edgecolor='0.8')

plt.title('2024 Election Predictions by State', fontsize=18)

legend_labels = {
    'darkblue': 'Safe Blue', 'blue': 'Likely Blue', 'lightblue': 'Lean Blue', 'skyblue': 'Tilt Blue',
    'darkred': 'Safe Red', 'red': 'Likely Red', 'lightcoral': 'Lean Red', 'salmon': 'Tilt Red'
}
for color, label in legend_labels.items():
    ax.plot([], [], marker="o", color=color, label=label, markersize=10)

ax.legend(title='Categories', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=14)

plt.tight_layout()

plt.show()
