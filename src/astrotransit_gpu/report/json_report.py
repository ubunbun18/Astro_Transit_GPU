import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def save_json_report(results, output_path):
    """Save results as a JSON file."""
    # Convert cupy to numpy if necessary
    serializable_results = {}
    for k, v in results.items():
        if hasattr(v, 'get'): # cupy.ndarray
            serializable_results[k] = v.get()
        else:
            serializable_results[k] = v
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, cls=NumpyEncoder, indent=2)
