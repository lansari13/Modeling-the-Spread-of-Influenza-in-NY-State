import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch_geometric.nn as geom_nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import random

import os
os.getcwd()

NUM_WEEKS = 33
all_data_objects = []
validation_data_objects = []
test_data_objects = []

def preprocess(filename):

  # Load the datasets
  merged_df_path = os.path.join('data', filename)
  clustering_df_path = os.path.join('data', 'updated_cluster_coefficient.csv')

  # Read the data
  merged_df = pd.read_csv(merged_df_path)
  clustering_df = pd.read_csv(clustering_df_path)

  # Merge clustering coefficients data with the main dataset
  merged_df = merged_df.merge(clustering_df, on='County', how='left')

  # Normalize only the Area
  area_scaler = MinMaxScaler()
  merged_df['Normalized_Area'] = area_scaler.fit_transform(merged_df[['Area_sq_miles']])

  # Sin-Cos Embedding for Week
  merged_df['Week_Sin'] = np.sin(2 * np.pi * merged_df['Flu Week'] / NUM_WEEKS)
  merged_df['Week_Cos'] = np.cos(2 * np.pi * merged_df['Flu Week'] / NUM_WEEKS)

  # Create lag features for 'Flu_Case_Count_per_100k' and 'Normalized Count'
  merged_df['Lag_1_Flu_Case_Count_per_100k'] = merged_df.groupby('County')['Flu_Case_Count_per_100k'].shift(1)
  merged_df['Lag_1_Normalized_Count'] = merged_df.groupby('County')['Normalized Count'].shift(1)

  # Create new features - Average Flu Case Count from 2 and 3 weeks ago
  merged_df['Avg_Flu_Case_Count_2_3_Weeks_Ago'] = (merged_df.groupby('County')['Flu_Case_Count_per_100k'].shift(2) + merged_df.groupby('County')['Flu_Case_Count_per_100k'].shift(3)) / 2

  # Create new feature - Average Normalized Count from 2 and 3 weeks ago
  merged_df['Avg_Normalized_Count_2_3_Weeks_Ago'] = (merged_df.groupby('County')['Normalized Count'].shift(2) + merged_df.groupby('County')['Normalized Count'].shift(3)) / 2

  # Fill NaN values resulted from shifting and missing values
  merged_df.fillna(0, inplace=True)

  # Selecting Features (X)
  feature_columns = ['Clustering Coefficient', 'Normalized_Area', 'Log_Population_Density', 'Week_Sin', 'Week_Cos', 'Lag_1_Flu_Case_Count_per_100k', 'Lag_1_Normalized_Count', 'Avg_Flu_Case_Count_2_3_Weeks_Ago', 'Avg_Normalized_Count_2_3_Weeks_Ago']

  # Load the adjacency list
  adjacency_list_path = os.path.join('data', 'Reduced_Adjacency_List.csv')
  adjacency_list_df = pd.read_csv(adjacency_list_path)

  # Convert county names to indices for the edge index
  county_to_index = {county: idx for idx, county in enumerate(merged_df['County'].unique())}

  # Create Edges
  edges = []
  for _, row in adjacency_list_df.iterrows():
      node_idx = county_to_index.get(row['County'])
      if node_idx is None:
          continue

      # Iterate over neighbors
      for neighbor in row[2:]:
          if pd.isna(neighbor):
              continue

          neighbor_idx = county_to_index.get(neighbor)
          if neighbor_idx is not None:
              edges.append((node_idx, neighbor_idx))
              edges.append((neighbor_idx, node_idx))

  # Convert the edges to a tensor
  edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

  weekly_data_objects = []
  for week in range(1, NUM_WEEKS + 1):

      # Filter the data for the current week
      week_data = merged_df[merged_df['Flu Week'] == week]

      # Prepare node features and target variable
      X = week_data[feature_columns]
      y = week_data['Count']

      X_tensor = torch.tensor(X.values, dtype=torch.float32)
      y_tensor = torch.tensor(y.values, dtype=torch.float32)

      # Create a data object for the current week
      data_object = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
      weekly_data_objects.append(data_object)

  return weekly_data_objects

# Load Datasets
file_names = ['Clean_09.csv', 'Clean_10.csv', 'Clean_12.csv', 'Clean_13.csv', 'Clean_14.csv', 'Clean_15.csv', 'Clean_19.csv', 'Clean_21.csv', 'Clean_22.csv']
feature_columns = None

for file_name in file_names:
    file_path = os.path.join('data', file_name)
    all_data_objects.append(preprocess(file_path))

file_names = ['Clean_16.csv', 'Clean_17.csv']
feature_columns = None

for file_name in file_names:
    file_path = os.path.join('data', file_name)
    validation_data_objects.append(preprocess(file_path))

file_names = ['Clean_18.csv']
feature_columns = None

for file_name in file_names:
    file_path = os.path.join('data', file_name)
    test_data_objects.append(preprocess(file_path))

class STANModel(nn.Module):
    def __init__(self, num_node_features, hidden_size, num_classes, heads):
        super(STANModel, self).__init__()
        self.gat1 = geom_nn.GATConv(num_node_features, hidden_size, heads=heads, concat=True)
        self.gat2 = geom_nn.GATConv(hidden_size*heads, hidden_size, heads=1, concat=False)
        #self.gat3 = geom_nn.GATConv(hidden_size*heads, hidden_size, heads=1, concat=False)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, data_list):
        week_embeddings = []
        for data in data_list:
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
              print("NaN or Inf in inputs")
            x, edge_index = data.x, data.edge_index
            x = self.gat1(x, edge_index)
            x = torch.relu(x)
            x = self.gat2(x, edge_index)
            x = torch.relu(x)
            #x = self.gat3(x, edge_index)
            #x = torch.relu(x)
            x = geom_nn.global_mean_pool(x, torch.zeros(data.num_nodes, dtype=torch.long, device=x.device))
            week_embeddings.append(x.unsqueeze(1))

        x_sequence = torch.cat(week_embeddings, dim=1)
        output_sequence, _ = self.gru(x_sequence)
        weekly_outputs = [self.out(output_sequence[:, i, :]) for i in range(output_sequence.size(1))]
        weekly_outputs = torch.stack(weekly_outputs, dim=1)

        return weekly_outputs

# Number of node features and output classes
num_node_features = 9
num_nodes = 62

# Hyperparameters
# 2 GAT Layers
hidden_size = 128
heads = 8
lr = 0.01
num_epochs = 200

# Initialize the STAN Model
model = STANModel(num_node_features, hidden_size, num_nodes, heads)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss function
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# TRAIN
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    random.shuffle(all_data_objects)

    for year_data in all_data_objects:
        year_data = [data_object.to(device) for data_object in year_data]

        # Prepare the targets for each week
        year_targets = torch.stack([data_object.y for data_object in year_data], dim=1).to(device)  # [batch_size, 33]

        # Check for NaN or Inf in targets
        if torch.isnan(year_targets).any() or torch.isinf(year_targets).any():
            print("NaN or Inf in targets")

        optimizer.zero_grad()
        output_sequence = model(year_data)  # [batch_size, 33, output_size]

        # Check for NaN or Inf in model output
        if torch.isnan(output_sequence).any() or torch.isinf(output_sequence).any():
            print("NaN or Inf in model output")

        loss = sum([criterion(output_sequence[:, i, :], year_targets[:, i]) for i in range(33)])

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("NaN or Inf in loss")

        # Compute loss for each week and sum up
        loss = sum([criterion(output_sequence[:, i, :], year_targets[:, i]) for i in range(33)])
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(all_data_objects)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')

# VALIDATE
model.eval()
total_loss = 0

with torch.no_grad():
    for year_data in validation_data_objects:
        year_data = [data_object.to(device) for data_object in year_data]
        
        # Prepare the targets for each week
        year_targets = torch.stack([data_object.y for data_object in year_data], dim=1).to(device)  # [batch_size, 33]
        output_sequence = model(year_data)  # [batch_size, 33, output_size]
        loss = sum([criterion(output_sequence[:, i, :], year_targets[:, i]) for i in range(33)])
        total_loss += loss.item()

        print(f'Validation Loss: {total_loss:.4f}')

# TEST
model.eval()
test_loss = 0
all_predictions = []
all_targets = []

with torch.no_grad():
    for year_data in test_data_objects:
        year_data = [data_object.to(device) for data_object in year_data]
        year_targets = torch.stack([data_object.y for data_object in year_data], dim=1).to(device)
        output_sequence = model(year_data)

        # Store predictions and actual targets
        predictions = output_sequence.squeeze().cpu().numpy()
        targets = year_targets.squeeze().cpu().numpy()
        all_predictions.append(predictions)
        all_targets.append(targets)

        # Calculate and accumulate loss
        loss = sum([criterion(output_sequence[:, i, :], year_targets[:, i]) for i in range(33)])
        total_loss += loss.item()

    print(f'Validation Loss: {total_loss:.4f}')
    # print("Sample Predictions:", all_predictions[0][0])
    # print("Sample Actuals:", all_targets[0][0])
    # print(len(all_predictions[0][0]))
    # print(len(all_targets[0][0]))

predictions_5 = all_predictions
actuals_5 = all_targets
type(all_predictions)

flat_list = [item for sublist in all_predictions for item in sublist]
flat_list_2 = [item for sublist in flat_list for item in sublist]
# print(f"Flat list (2D): {flat_list_2}")
print(f"Sum: {sum(flat_list_2)}")

if len(all_predictions) > 0 and len(all_targets) > 0:
    predictions = all_predictions[0]
    #actuals = all_targets[0]
    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predicted', color='blue')
    #plt.plot(actuals, label='Actual', color='orange')
    plt.title('Predicted Flu Cases 2018 - (all 62 Counties)')
    plt.xlabel('Flu Week')
    plt.ylabel('Flu Case Count')
    #plt.legend()
    plt.show()

# Calculate the total count for each county
total_counts_pred = np.sum(all_predictions[0], axis=0)
total_counts_target = np.sum(all_targets[0], axis=1)

counties = list(range(1, 63))

plt.figure(figsize=(15, 8))

# Create bar plots
width = 0.35
plt.bar([x - width/2 for x in counties], total_counts_pred, width=width, label='Predicted', color='blue')
plt.bar([x + width/2 for x in counties], total_counts_target, width=width, label='Actual', color='orange')

plt.xlabel('County')
plt.ylabel('Total Count')
plt.title('Predicted and Actual Total Counts per County 2018')
plt.xticks(counties)  # Set the x-ticks to be the county numbers
plt.legend()

plt.tight_layout()
plt.show()
