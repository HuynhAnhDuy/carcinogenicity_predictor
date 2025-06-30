import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter
from sklearn.ensemble import RandomForestClassifier

# Hàm tính EState
def calculate_estate(df, smiles_col):
    def get_estate(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * 79
            values = Fingerprinter.FingerprintMol(mol)[0]
            return [round(v, 3) for v in values]
        except Exception:
            return [None] * 79
    est_df = df[smiles_col].apply(get_estate).apply(pd.Series)
    est_df.columns = [f"EState_{i+1}" for i in range(79)]
    return est_df

# Đọc dữ liệu
df = pd.read_csv("carcinogen_x_train.csv")  # cần cột 'canonical_smiles' và 'Label'

# Tính estate fingerprint
estate_df = calculate_estate(df, "canonical_smiles")

# Nối lại thành một DataFrame duy nhất
df_final = pd.concat([df, estate_df], axis=1)

# Loại bỏ hàng có giá trị None
df_final = df_final.dropna()

# Tạo X, y
X = df_final[[f"EState_{i+1}" for i in range(79)]].values
y = df_final["Label"].values

# Huấn luyện mô hình
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Lưu mô hình
with open("rf_estate_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Đã lưu mô hình Random Forest với EState vào 'rf_estate_model.pkl'")
