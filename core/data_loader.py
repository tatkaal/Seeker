# --- 1. Data Loading and Preprocessing ---
import json
import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self, sample_size=None):
        """Loads data from JSON file, potentially sampling."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            if sample_size:
                return pd.DataFrame(data).sample(n=min(sample_size, len(data)), random_state=42)
            return pd.DataFrame(data)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.filepath}")
            # Fallback to the 10 samples if main file not found, for demonstration
            print("Loading inline sample data for demonstration.")
            samples_json_string = """
            [{"id":38885743,"title":"SYSTEMS TRAINING & SUPPORT SPECIALIST, LJ Hooker Corporate","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38885221,"title":"Receptionist","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38918237,"title":"Sales Person \/ Forklift Driver \/ Warehouse","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38872849,"title":"B737 First Officers","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38957774,"title":"Marine Technician Submariner","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38835993,"title":"QA Manager","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38952335,"title":"Real Estate Sales","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38900437,"title":"Maintenance Project Engineer - Transport \/ Kaimatai Puhanga Kaupapa Tautiaki","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38859133,"title":"Heavy Vehicle Mechanic - Narellan","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}},
            {"id":38946240,"title":"Contracts Administrator - Fit-out\/Construction","abstract":"...","content":"<HTML>...<\/HTML>","metadata":{}}]
            """ # Truncated for brevity, assume full structures from user prompt for these
            return pd.DataFrame(json.loads(samples_json_string.replace("...", "")))
