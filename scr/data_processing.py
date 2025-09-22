"""
Data cleaning and preprocessing functions for job skills analysis.
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict
from bs4 import BeautifulSoup
from numpy import nan, NAN
import glob

TECH_SKILLS = [
    # Programming Languages
    'python', 'r', 'sql', 'java', 'scala', 'julia', 'c++', 'c#', 'javascript', 'typescript', 'sas', 'matlab', 'spss',
    # Big Data & Cloud
    'spark', 'hadoop', 'hive', 'kafka', 'airflow', 'aws', 'azure', 'gcp', 'google cloud', 'snowflake', 'redshift', 'bigquery', 'databricks', 'docker', 'kubernetes',
    # Databases
    'postgresql', 'mysql', 'mongodb', 'cassandra', 'redis', 'elasticsearch', 'dynamodb', 'oracle',
    # ML & AI Frameworks
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'mxnet', 'h2o', 'pytorch', 'xgboost', 'lightgbm', 'catboost',
    # ML & AI Concepts
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision', 'reinforcement learning', 'time series', 'recommendation systems', 'llm', 'large language model',
    # Data Engineering
    'etl', 'elt', 'data pipeline', 'data modeling', 'data warehouse', 'data lake', 'dbt',
    # BI & Visualization
    'tableau', 'power bi', 'looker', 'quicksight', 'matplotlib', 'seaborn', 'plotly', 'd3.js',
    # Tools & Other
    'excel', 'git', 'jupyter', 'linux', 'bash', 'shell', 'ci/cd'
]

def get_job_description(description):
    """Scrape the full job description from HTML content."""
    if pd.isna(description) or not isinstance(description, str):
        return ""
    
    try:
        soup = BeautifulSoup(description, 'html.parser')
        
        # Different sites have different structures. We try common patterns.
        description_text = ""
        
        # Pattern 1: Greenhouse - often in a div with specific section
        greenhouse_desc = soup.find('div', {'id': 'content'})
        if greenhouse_desc:
            description_text = greenhouse_desc.get_text(separator=' ', strip=True)
        
        # Pattern 2: Lever - often in a div with class 'content'
        if not description_text:
            lever_desc = soup.find('div', class_='content')
            if lever_desc:
                description_text = lever_desc.get_text(separator=' ', strip=True)
        
        # Pattern 3: Generic fallback - get all text
        if not description_text:
            description_text = soup.get_text(separator=' ', strip=True)
        
        return description_text[:10000]  # Return first 10k chars to avoid huge texts
    
    except Exception as e:
        print(f"Error parsing HTML description: {e}")
        return description  # Return original if parsing fails

def extract_skills(description_text):
    """Extract mentioned skills from a job description string."""
    if not description_text or pd.isna(description_text):
        return NAN
    
    # Convert to lowercase for case-insensitive matching
    text_lower = description_text.lower()
    found_skills = []
    
    for skill in TECH_SKILLS:
        # Use regex to find whole words to avoid partial matches
        # e.g., doesn't match 'java' in 'javascript'
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)
    
    return found_skills if found_skills else NAN

def load_and_clean_data(filepath):
    """
    Load raw job data and perform basic cleaning.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    DataFrame: Cleaned job data
    """
    try:
        df = pd.read_csv(filepath)
        
        # Basic cleaning
        df = df.dropna(subset=['description']).copy()
        df['description'] = df['description'].astype(str)
        
        return df
    
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def process_single_file(filepath):
    """
    Process a single CSV file: load, extract skills, and return processed data.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    DataFrame: Processed data with skills column
    """
    print(f"Processing: {os.path.basename(filepath)}")
    
    # Load data
    df = load_and_clean_data(filepath)
    if df.empty:
        print(f"  No data found in {filepath}")
        return pd.DataFrame()
    
    print(f"  Found {len(df)} rows")
    
    # Extract text from HTML descriptions
    print("  Extracting text from HTML descriptions...")
    df['description'] = df['description'].apply(get_job_description)
    df['description'] = df['description'].apply(get_job_description)
    
    # Extract skills
    print("  Extracting skills from descriptions...")
    df['skills'] = df['description'].apply(extract_skills)
    
    # Remove rows with no skills found
    initial_count = len(df)
    df = df[df['skills'].notna()].copy()
    print(f"  Found skills in {len(df)} out of {initial_count} rows")
    
    # Add source file information
    df['source_file'] = os.path.basename(filepath)
    try:
        df = df.drop(columns=['ats'])
    except KeyError:
        pass  # If 'ats' column doesn't exist, ignore
    
    return df

# Enhanced title categorization with experience levels and comprehensive role types
def categorize_title(title):
    if pd.isna(title):
        return 'Unknown'
    
    title_lower = str(title).lower()
    
    # Experience level detection
    experience_level = 'Mid'  # Default to Mid level
    
    if re.search(r'\b(junior|jr|entry.level|associate)\b', title_lower):
        experience_level = 'Junior'
    elif re.search(r'\b(senior|sr|lead|principal|staff|manager|director|head of)\b', title_lower):
        experience_level = 'Senior'
    
    # Role type detection
    if re.search(r'\bdata.*scientist\b', title_lower):
        role_type = 'Data Scientist'
    elif re.search(r'\bdata.*analyst\b', title_lower):
        role_type = 'Data Analyst'
    elif re.search(r'\bdata.*engineer\b', title_lower):
        role_type = 'Data Engineer'
    elif re.search(r'\bmachine.*learning.*engineer\b|\bml.*engineer\b', title_lower):
        role_type = 'Machine Learning Engineer'
    elif re.search(r'\bai.*engineer\b|\bartificial.*intelligence.*engineer\b', title_lower):
        role_type = 'AI Engineer'
    elif re.search(r'\bbusiness.*intelligence.*analyst\b|\bbi.*analyst\b', title_lower):
        role_type = 'Business Intelligence Analyst'
    elif re.search(r'\bdata.*architect\b', title_lower):
        role_type = 'Data Architect'
    elif re.search(r'\bdata.*science\b', title_lower) and not re.search(r'\b(data scientist|data science manager)\b', title_lower):
        role_type = 'Data Science General'
    elif re.search(r'\banalytics.*engineer\b', title_lower):
        role_type = 'Analytics Engineer'
    elif re.search(r'\bresearch.*scientist\b', title_lower):
        role_type = 'Research Scientist'
    elif re.search(r'\bquantitative.*analyst\b|\bquant.*analyst\b', title_lower):
        role_type = 'Quantitative Analyst'
    else:
        # Fallback: check for any data-related terms
        if any(term in title_lower for term in ['data', 'analytics', 'machine learning', 'ai', 'business intelligence']):
            role_type = 'Other Data Role'
        else:
            role_type = 'Non-Data Role'
    
    return f"{experience_level} {role_type}"

def process_all_csv_files(data_dir='data/raw', output_file='data/processed/cleaned_job_data.csv'):
    """
    Process all CSV files in the data directory and create a consolidated dataset.
    
    Parameters:
    data_dir (str): Directory containing CSV files
    output_file (str): Path for output consolidated CSV
    
    Returns:
    DataFrame: Consolidated dataset with all processed data
    """
    # Create processed directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each file
    all_dataframes = []
    for filepath in csv_files:
        processed_df = process_single_file(filepath)
        if not processed_df.empty:
            all_dataframes.append(processed_df)
    
    if not all_dataframes:
        print("No data processed from any files")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    print("Concatenating all processed data...")
    consolidated_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Basic cleaning on consolidated data
    consolidated_df = consolidated_df.dropna(subset=['skills']).copy()
    
    # Ensure skills column is properly formatted
    consolidated_df['skills'] = consolidated_df['skills'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    
    # Remove duplicates based on description text (if applicable)
    if 'description' in consolidated_df.columns:
        consolidated_df = consolidated_df.drop_duplicates(subset=['description'])

    consolidated_df['title_category'] = consolidated_df['title'].apply(categorize_title)
    
    consolidated_df = consolidated_df[~consolidated_df['title_category'].isin(['Senior Non-Data Role','Mid Non-Data Role','Junior Non-Data Role']) ]
    print(f"Final consolidated dataset: {len(consolidated_df)} rows")
    consolidated_df = consolidated_df[['company','title','title_category','location','url','description','skills','source']]
    # Save to CSV
    print(f"Saving consolidated data to {output_file}")

    consolidated_df.to_csv(output_file, index=False)
    
    # Also save a summary report
    save_processing_summary(consolidated_df, output_file.replace('.csv', '_summary.txt'))
    
    return consolidated_df

def save_processing_summary(df, summary_file):
    """Save a summary of the processing results."""
    with open(summary_file, 'w') as f:
        f.write("JOB DATA PROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total jobs processed: {len(df)}\n")
        
        f.write("Skills distribution:\n")
        # Count skills
        skill_counts = {}
        for skills_list in df['skills']:
            for skill in skills_list:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Sort by frequency
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        
        for skill, count in sorted_skills[:20]:  # Top 20 skills
            f.write(f"  {skill}: {count} jobs ({count/len(df)*100:.1f}%)\n")
        
