from evidently.report import Report
from evidently.metrics import DataDriftTable, ClassificationQuality
import pandas as pd
import numpy as np

class DriftDetector:
    def __init__(self, reference_data):
        self.reference = reference_data
        self.report = Report(metrics=[
            DataDriftTable(),
            ClassificationQuality()
        ])
    
    def check_drift(self, current_data):
        self.report.run(reference_data=self.reference, current_data=current_data)
        return self.report.as_dict()
    
    def generate_alert(self, drift_metrics, threshold=0.2):
        if drift_metrics['data_drift']['dataset_drift']:
            return "CRITICAL: Significant data drift detected"
        if drift_metrics['classification']['f1'] < threshold:
            return f"WARNING: Model performance dropped (F1={drift_metrics['classification']['f1']:.3f})"
        return None

# Sample usage
if __name__ == "__main__":
    # Load reference data (from training)
    ref_data = pd.read_csv("reference_data.csv")
    
    detector = DriftDetector(ref_data)
    current_data = pd.read_csv("current_data.csv")
    
    results = detector.check_drift(current_data)
    alert = detector.generate_alert(results)
    if alert:
        print(alert)