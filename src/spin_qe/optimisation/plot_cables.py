import pandas as pd
import matplotlib.pyplot as plt

# Function to load CSV data
def load_csv_data(filename):
    return pd.read_csv(filename)

# Filenames
filenames = [
    'energy_vs_temperature_conduction_COAXcable.csv',
    'energy_vs_temperature_conduction_COAXScable.csv',
    'energy_vs_temperature_conduction_MScable.csv',
    'energy_vs_temperature_conduction_MSScable.csv'
]

# Labels for the cables
cable_labels = [
    'COAXcable',
    'COAXScable',
    'MScable',
    'MSScable'
]

# Colors for plotting
colors = ['blue', 'green', 'red', 'orange']

# Plot
plt.figure(figsize=(10, 6))

for idx, (filename, label) in enumerate(zip(filenames, cable_labels)):
    # Load the data
    data = load_csv_data(filename)

    # Extract columns
    Tqb = data['Tqb']
    energyCarnot = data['energyCarnot']
    energySS = data['energySS']

    # Plot Carnot energy (solid line)
    plt.plot(Tqb, energyCarnot, linewidth=2.5, label=f'Carnot ({label})', color=colors[idx])

    # Plot energySS (dashed line)
    plt.plot(Tqb, energySS, linestyle='--', linewidth=2.5, label=f'Small System ({label})', color=colors[idx])

# Formatting the plot
plt.xlabel(r'Qubit temperature $T_q$ (K)', fontsize=16)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'$E_{cond}$ (Joules)', fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.legend(fontsize=12)
plt.title('Energy vs Temperature for Different Cable Types', fontsize=16)
plt.tight_layout()

# Save the plot
plt.savefig('energy_vs_temperature_cables.svg', format='svg', bbox_inches='tight')
plt.savefig('energy_vs_temperature_cables.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()
