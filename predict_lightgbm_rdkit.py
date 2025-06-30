import numpy as np
import pickle
from rdkit import Chem

# ===== Hàm tính RDKit fingerprint từ 1 SMILES =====
def smiles_to_rdkit_fp(smiles, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((nBits,), dtype=np.uint8)
        fp = Chem.RDKFingerprint(mol, fpSize=nBits)
        return np.array([int(x) for x in fp.ToBitString()], dtype=np.uint8)
    except Exception:
        return np.zeros((nBits,), dtype=np.uint8)

# ===== Hàm dự đoán xác suất carcinogen =====
def predict_from_smiles(smiles, model_path="lightgbm_model.pkl"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    fp = smiles_to_rdkit_fp(smiles).reshape(1, -1)
    prob = model.predict_proba(fp)[0][1]  # Xác suất là nhãn 1
    return prob
