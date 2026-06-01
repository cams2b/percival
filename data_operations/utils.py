import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


def transform_to_json(df, json_output_path):    
    # Ensure study_date is treated as string for sorting and formatting
    df['study_date'] = df['study_date'].astype(str)
    
    # Sort by patient and date (descending) to prepare for reverse chronological order
    df = df.sort_values(by=['patient_id', 'study_date'], ascending=[True, False])
    
    json_data = []
    
    # Group by patient
    for patient_id, p_group in df.groupby('patient_id', sort=False):
        patient_entry = {
            "patient_id": str(patient_id),
            "studies": []
        }
        
        # Group by study (visit_id) within patient
        # Using sort=False to maintain the date sorting we did earlier
        for visit_id, s_group in p_group.groupby('visit_id', sort=False):
            study_date = s_group['study_date'].iloc[0]
            
            # Format date to YYYY-MM-DD if it's YYYYMMDD
            if len(study_date) == 8:
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
            
            study_entry = {
                "study_id": str(visit_id),
                "study_date": study_date,
                "progression_path": "", # TODO: Add progression files here
                "scans": []
            }
            
            # Add scans for this study
            for _, row in s_group.iterrows():
                scan_entry = {
                    "scan_id": str(row['scan_id']),
                    "image_path": row['image_path'],
                    "report_path": row['report_path']
                }
                study_entry["scans"].append(scan_entry)
            
            patient_entry["studies"].append(study_entry)
        
        json_data.append(patient_entry)
    
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"Successfully transformed {len(df)} rows into {len(json_data)} patients at {json_output_path}")