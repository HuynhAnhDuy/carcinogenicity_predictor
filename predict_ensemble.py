from rdkit import Chem
from predict_lightgbm_rdkit import predict_from_smiles as pred_lgbm
from predict_rf_estate import predict_rf_from_smiles as pred_rf
from predict_bilstm_maccs import predict_bilstm_from_smiles as pred_bilstm

def canonicalize_smiles(smiles: str) -> str:
    """Convert input SMILES to canonical form."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def predict_ensemble(smiles):
    """Run ensemble prediction from three models."""
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {"error": "❌ Invalid SMILES format."}

    try:
        prob_lgbm = pred_lgbm(canonical)
        prob_rf = pred_rf(canonical)
        prob_bilstm = pred_bilstm(canonical)
    except Exception as e:
        return {"error": f"❌ Prediction error: {str(e)}"}

    average_prob = (prob_lgbm + prob_rf + prob_bilstm) / 3
    label = "carcinogen" if average_prob > 0.5 else "non-carcinogen"

    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "prob_lightgbm": prob_lgbm,
        "prob_rf": prob_rf,
        "prob_bilstm": prob_bilstm,
        "average_probability": average_prob,
        "predicted_label": label
    }

# === Demo ===
if __name__ == "__main__":
    smiles_input = "NC(=O)CCCCC(N)=O"  # Replace with any SMILES you want to test
    result = predict_ensemble(smiles_input)

    if "error" in result:
        print(result["error"])
    else:
        print(f"Input SMILES:             {smiles_input}")
        print(f"LightGBM Probability:     {result['prob_lightgbm']:.4f}")
        print(f"RF EState Probability:    {result['prob_rf']:.4f}")
        print(f"BiLSTM MACCS Probability: {result['prob_bilstm']:.4f}")
        print(f"Average Probability:      {result['average_probability']:.4f}")
        print(f"Prediction:               {result['predicted_label']}")
