import pandas as pd

# Load the CSV file into a DataFrame

df = pd.read_csv(
    "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/guardian/knn_model/test_1conv_7117532915-300-4.csv"
)  # Replace 'path_to_your_file.csv' with the actual file path
# df = pd.read_csv(
#     "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/guardian/knn_model/test_6362141245-300-1.csv"
# )  # Replace 'path_to_your_file.csv' with the actual file path
# df = pd.read_csv(
#     "H:/1.Deakin university/Python/13_10_2023_My_Project_1/guardian_paper_my_version/data/guardian/knn_model/test_default2209158907-300.csv"
# )  # Replace 'path_to_your_file.csv' with the actual file path


# Skip the first column
df = df.iloc[:, 1:]

# Skip the first row and start processing from the second row
df = df.iloc[1:]


# Function to label each row
def label_row(row):
    return "attack" if (row > 0.520).sum() > 2 else "normal"


# Initialize counters for each label
total_attack = 0
total_normal = 0

# Process each batch of 10 rows
for start_row in range(0, len(df), 10):
    # Select a batch of 10 rows
    batch = df.iloc[start_row : start_row + 10].copy()

    # If the batch is empty, break the loop
    if batch.empty:
        break

    # Apply the label_row function to each row in the batch
    batch["Label"] = batch.apply(label_row, axis=1)

    # Determine the final decision for the batch
    final_decision = "attack" if (batch["Label"] == "attack").sum() > 2 else "normal"

    # Update counters based on the final decision
    if final_decision == "attack":
        total_attack += 1
    else:
        total_normal += 1

    # Print the final decision for the current batch
    # print(f"Batch starting at row {start_row + 2}: {final_decision}")

# Print the report of total batches labeled as "normal" and "attack"
print(f"\nTotal batches labeled as 'normal': {total_normal}")
print(f"Total batches labeled as 'attack': {total_attack}")
