import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam

# === Tính MACCS fingerprints ===
def calculate_maccs(df, smiles_col):
    def get_maccs(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = MACCSkeys.GenMACCSKeys(mol)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * 167
    maccs_df = df[smiles_col].apply(get_maccs).apply(pd.Series)
    maccs_df.columns = [f"MACCS{i}" for i in range(167)]
    return maccs_df

# === Xây dựng mô hình BiLSTM ===
def build_model(input_dim):
    model = Sequential([
        Input(shape=(1, input_dim)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(100, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# === Đọc dữ liệu ===
df = pd.read_csv("carcinogen_x_train.csv")  # Cột bắt buộc: 'canonical_smiles' và 'Label'

# === Tính MACCS ===
maccs_df = calculate_maccs(df, "canonical_smiles")
df_final = pd.concat([df, maccs_df], axis=1).dropna()

# === Tạo dữ liệu đầu vào ===
X = df_final[[f"MACCS{i}" for i in range(167)]].values
y = df_final["Label"].values

# === Reshape dữ liệu cho BiLSTM ===
X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, timesteps, features)

# === Huấn luyện ===
model = build_model(input_dim=167)
model.fit(X_reshaped, y, batch_size=32, epochs=20, validation_split=0.2)

# === Lưu mô hình ===
model.save("bilstm_model.keras")  # Tạo file .keras theo định dạng zip mới
print("✅ Saved BiLSTM model to 'bilstm_model.keras'")
