import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('Dataset.csv')

# Clean column names
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

# Fill missing categorical values with mode
categorical_cols = ['gender', 'married', 'dependents', 'education', 'self_employed', 'property_area']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical features
encoders = {}
for col in categorical_cols + ['loan_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Fill missing numerical values
df['loanamount'] = SimpleImputer(strategy='median').fit_transform(df[['loanamount']])
df['loan_amount_term'] = df['loan_amount_term'].fillna(360)
df['credit_score'] = SimpleImputer(strategy='mean').fit_transform(df[['credit_score']])

# Double-check and impute any remaining numerical NaNs
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col] = SimpleImputer(strategy='mean').fit_transform(df[[col]])

# Drop unnecessary columns
if 'loan_id' in df.columns:
    df.drop(columns=['loan_id'], inplace=True)

# Check for remaining NaNs
missing_summary = df.isnull().sum()
print("Missing values after imputation:\n", missing_summary[missing_summary > 0])
assert df.isnull().sum().sum() == 0, "Still contains NaNs"

# Prepare features and labels
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add noise to training data
noise = np.random.normal(0, 0.65 , X_train.shape)  # Increased noise std from 0.65 to 0.7
X_train = X_train + noise

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define simpler model with higher dropout
class LoanApprovalModel(nn.Module):
    def __init__(self, input_dim):
        super(LoanApprovalModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)  # Reduced neurons
        self.dropout1 = nn.Dropout(0.6)      # High dropout
        self.fc2 = nn.Linear(32, 8)          # Reduced neurons
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Model setup
input_dim = X_train.shape[1]
model = LoanApprovalModel(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with 100 epochs
def train_model(model, X_train, y_train, X_test, y_test, num_epochs=100, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Train
train_model(model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

# Evaluate
model.eval()
with torch.no_grad():
    y_pred = (model(X_test_tensor) > 0.5).int()
    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred.numpy())
    print(f"PyTorch Model Accuracy: {accuracy:.4f}")

# Save model
torch.save(model.state_dict(), 'loan_approval_model.pth')