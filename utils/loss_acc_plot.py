import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# List of CSV file names


csv_files = [r'C:\Users\hjl15\Desktop\SegFormer_validation_acc.csv',
             r'C:\Users\hjl15\Desktop\SegFormer_train_acc.csv',
             r'C:\Users\hjl15\Desktop\DFormer_validation_acc.csv',
             r'C:\Users\hjl15\Desktop\DFormer_train_acc.csv']  # Replace with your file names





# Iterate over each CSV file
sns.set(style="darkgrid")

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate over each CSV file
for file in csv_files:
    # Read CSV file into a DataFrame
    df = pd.read_csv(file, header=None)

    # Extract the data from the DataFrame
    data = df.values.flatten()

    # Plot the data
    ax.plot(data, label=file)

# Set the figure title and labels
ax.set_title("Data from CSV Files")
ax.set_xlabel("Index")
ax.set_ylabel("Value")

# Add a legend
ax.legend()

# Adjust the layout
plt.tight_layout()

# Show the figure
plt.show()
