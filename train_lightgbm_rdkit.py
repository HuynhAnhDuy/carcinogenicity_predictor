import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from lightgbm import LGBMClassifier
import pickle

# ===== Hàm tính RDKit fingerprint (bit vector 2048 chiều) =====
def calculate_rdkit(df, smiles_col, nBits=2048):
    def get_rdkit(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return [None] * nBits
            fp = Chem.RDKFingerprint(mol, fpSize=nBits)
            return [int(x) for x in fp.ToBitString()]
        except Exception:
            return [None] * nBits
    rdk_df = df[smiles_col].apply(get_rdkit).apply(pd.Series)
    rdk_df.columns = [f"RDKit_{i+1}" for i in range(nBits)]
    return rdk_df

# ===== Đọc dữ liệu =====
df = pd.read_csv("carcinogen_x_train.csv")  # Gồm 'canonical_smiles' và 'Label'

# ===== Tính RDKit fingerprint =====
rdk_df = calculate_rdkit(df, "canonical_smiles")

# ===== Ghép với nhãn và xử lý thiếu =====
full_df = pd.concat([rdk_df, df["Label"]], axis=1)
full_df = full_df.dropna()

# ===== Tạo X, y =====
X = full_df.drop(columns=["Label"]).values.astype(np.uint8)
y = full_df["Label"].values

# ===== Huấn luyện mô hình LightGBM =====
model = LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ===== Lưu mô hình =====
with open("lightgbm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Mô hình LightGBM đã được huấn luyện và lưu vào 'lightgbm_model.pkl'")
