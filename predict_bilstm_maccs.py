from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np

# === Load model ===
try:
    model = load_model("bilstm_model.keras")  # File định dạng .keras
except Exception as e:
    raise RuntimeError(f"❌ Failed to load BiLSTM model: {e}")

# === Convert SMILES to MACCS ===
def smiles_to_maccs(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array([int(x) for x in fp.ToBitString()])
    except Exception as e:
        print(f"⚠️ MACCS conversion failed for '{smiles}': {e}")
        return np.zeros((167,))

# === Predict from SMILES ===
def predict_bilstm_from_smiles(smiles):
    fp = smiles_to_maccs(smiles).reshape((1, 1, 167))
    try:
        prob = model.predict(fp, verbose=0)[0][0]
        return float(prob)
    except Exception as e:
        raise RuntimeError(f"❌ Prediction failed: {e}")
