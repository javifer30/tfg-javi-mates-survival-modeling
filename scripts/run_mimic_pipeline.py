import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

XMI_ICU_PATH = PROJECT_ROOT / "src" / "models_references" / "XMI-ICU"
DATA_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mimic_extraction"

# Add XMI-ICU to path for eICU_preprocessing imports
sys.path.append(str(XMI_ICU_PATH))

# Import local extraction script
from src.data import mimic_direct_extraction
from src.data import mimic_timeseries_sparse

def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    print("=== Starting MIMIC-IV Pipeline Replication ===")
    
    # 1. Extraction
    print("\n--- Step 1: Direct Extraction from Raw CSVs ---")
    # Check if files already exist to skip? Or always run? Let's run.
    #labels = mimic_direct_extraction.generate_labels()
    #mimic_direct_extraction.generate_flat_features(labels)
    #mimic_direct_extraction.generate_timeseries(labels)
    
    print("\n--- Step 1 Complete ---")
    
    # 2. Load MIMIC Preprocessing Modules
    print("\nLoading external modules from XMI-ICU...")
    
    fl_script_path = XMI_ICU_PATH / "MIMIC_IV-preprocessing" / "flat_and_labels.py"
    
    try:
        mimic_flat_labels = import_module_from_path("mimic_flat_labels", fl_script_path)
    except Exception as e:
        print(f"Error loading modules: {e}")
        print("Ensure 'MIMIC_IV-preprocessing' is accessible via sys.path.")
        return

    # 3. Run Preprocessing
    print("\n--- Step 2: Time Series Preprocessing (Efficient Mode) ---")
    # Using the local efficient implementation
    mimic_timeseries_sparse.main(str(DATA_OUTPUT_PATH) + "/")
    
    print("\n--- Step 3: Flat and Labels Preprocessing ---")
    # This generates preprocessed_flat.csv and preprocessed_labels.csv
    mimic_flat_labels.flat_and_labels_main(str(DATA_OUTPUT_PATH) + "/")
    
    print("\n=== Pipeline Complete ===")
    print(f"Processed data available in: {DATA_OUTPUT_PATH}")
    print("Files generated:")
    print(" - preprocessed_flat.csv")
    print(" - preprocessed_labels.csv")
    print(" - preprocessed_timeseries.csv")

if __name__ == "__main__":
    main()
