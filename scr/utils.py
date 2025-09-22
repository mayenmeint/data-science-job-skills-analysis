"""
Utility functions for the skills analysis project.
"""

import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any
from scr.skill_analysis import analyze_skill_frequency

def save_analysis_results(results: Dict[str, Any], filename: str):
    """
    Save analysis results to JSON file.
    
    Parameters:
    results (dict): Analysis results to save
    filename (str): Output filename
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_analysis_results(filename: str) -> Dict[str, Any]:
    """
    Load analysis results from JSON file.
    
    Parameters:
    filename (str): Input filename
    
    Returns:
    dict: Loaded analysis results
    """
    with open(filename, 'r') as f:
        return json.load(f)

def calculate_confidence_intervals(df, confidence=0.95):
    """
    Calculate confidence intervals for skill frequencies.
    
    Parameters:
    df (DataFrame): Job data
    confidence (float): Confidence level (0.95 for 95%)
    
    Returns:
    DataFrame: Skill frequencies with confidence intervals
    """
    from scipy import stats
    
    skill_freq = analyze_skill_frequency(df)
    total_jobs = len(df)
    
    skill_freq['proportion'] = skill_freq['frequency'] / total_jobs
    skill_freq['std_error'] = np.sqrt(skill_freq['proportion'] * (1 - skill_freq['proportion']) / total_jobs)
    
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    skill_freq['ci_lower'] = skill_freq['proportion'] - z_score * skill_freq['std_error']
    skill_freq['ci_upper'] = skill_freq['proportion'] + z_score * skill_freq['std_error']
    
    return skill_freq

def export_to_excel(analysis_results: Dict[str, pd.DataFrame], filename: str):
    """
    Export multiple analysis results to Excel file with different sheets.
    
    Parameters:
    analysis_results (dict): Dictionary of {sheet_name: DataFrame}
    filename (str): Output Excel filename
    """
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for sheet_name, df in analysis_results.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)