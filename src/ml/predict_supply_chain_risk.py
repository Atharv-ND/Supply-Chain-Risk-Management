import pickle
import numpy as np

class SupplyChainRiskPredictor:
    def __init__(self, classifier_path='data/models/phase9_best_model.pkl', regressor_path='data/models/phase9_delay_regressor.pkl'):
        """
        Loads the saved classification and regression pipelines.
        """
        with open(classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
            
        with open(regressor_path, 'rb') as f:
            self.delay_regressor = pickle.load(f)
            
    def predict_shipment(self, features_df, threshold=0.20):
        """
        Performs conditional two-stage prediction on the provided features dataframe.
        The dataframe must contain exactly the 41 approved ML predictors.
        
        Args:
            features_df (pd.DataFrame): The input features (single row or multiple rows).
            threshold (float): The operational threshold for predicting a late shipment.
            
        Returns:
            list of dict: For each shipment, returns a dict with late_probability, 
                          late_prediction, predicted_delay_days, and risk_level.
        """
        # Ensure correct column count conceptually
        # assert len(features_df.columns) == 41, "Input must have exactly 41 features."
        
        # 1. Model 1 - Classifier Predicts Probability
        late_probs = self.classifier.predict_proba(features_df)[:, 1]
        
        results = []
        for i, prob in enumerate(late_probs):
            if prob >= threshold:
                late_pred = 1
                # Execute Model 2 only for shipments flagged as late risk
                # Extract single row as DataFrame to maintain feature names for the pipeline
                single_row = features_df.iloc[[i]]
                pred_delay = self.delay_regressor.predict(single_row)[0]
                pred_delay = max(0.0, float(pred_delay))
            else:
                late_pred = 0
                # Model 2 does not execute
                pred_delay = 0.0
                
            # Basic risk level mapping
            if pred_delay == 0:
                risk_level = "LOW"
            elif pred_delay <= 2:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
                
            results.append({
                "late_probability": float(prob),
                "late_prediction": late_pred,
                "predicted_delay_days": round(pred_delay, 2),
                "risk_level": risk_level
            })
            
        return results
