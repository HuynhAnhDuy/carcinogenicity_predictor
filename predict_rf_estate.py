import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem.EState import Fingerprinter

def smiles_to_estate(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(79)
        values = Fingerprinter.FingerprintMol(mol)[0]
        return np.array(values)
    except Exception:
        return np.zeros(79)

def predict_rf_from_smiles(smiles, model_path="rf_estate_model.pkl"):
    fp = smiles_to_estate(smiles).reshape(1, -1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    prob = model.predict_proba(fp)[0][1]
    return prob
