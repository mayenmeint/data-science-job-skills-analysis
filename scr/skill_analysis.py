"""
Core functions for analyzing job skills data.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Union

def get_skill_list(skills: Union[str, list]) -> List[str]:
    """
    Convert skills string to list format.
    
    Parameters:
    skills (str or list): Skills data in string or list format
    
    Returns:
    List[str]: List of skills
    """
    try:
        if isinstance(skills, str):
            # Handle string format like "['python','sql','machine learning']"
            return skills[1:-1].replace(' ', '').replace("'", '').split(',')
        elif isinstance(skills, list):
            return skills
        else:
            return []
    except Exception as e:
        print(f"Error converting skills to list: {e}")
        return []

def analyze_skill_frequency(df: pd.DataFrame, skills_column: str = 'skills') -> pd.DataFrame:
    """
    Analyzes how frequently each skill appears across all job postings.
    
    Parameters:
    df (DataFrame): DataFrame containing job data
    skills_column (str): Name of the column containing skills
    
    Returns:
    DataFrame: Skills with their frequency counts and percentages
    """
    skill_counts = {}
    total_jobs = len(df)
    
    for skills_list in df[skills_column]:
        processed_skills = get_skill_list(skills_list)
        for skill in processed_skills:
            if skill:  # Skip empty strings
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    # Create DataFrame with frequencies and percentages
    skill_freq_df = pd.DataFrame(list(skill_counts.items()), 
                                columns=['skill', 'frequency'])
    skill_freq_df['percentage'] = (skill_freq_df['frequency'] / total_jobs * 100).round(2)
    
    return skill_freq_df.sort_values('frequency', ascending=False)

def analyze_skills_by_title(df: pd.DataFrame, 
                          title_column: str = 'title_category', 
                          skills_column: str = 'skills') -> pd.DataFrame:
    """
    Analyzes skill requirements grouped by job title.
    
    Parameters:
    df (DataFrame): DataFrame containing title and skills data
    title_column (str): Name of the column containing job titles
    skills_column (str): Name of the column containing skills
    
    Returns:
    DataFrame: Skills matrix with titles as rows and skills as columns
    """
    title_skills = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        title = row[title_column]
        skills_list = get_skill_list(row[skills_column])
        
        for skill in skills_list:
            if skill:  # Skip empty strings
                title_skills[title][skill] += 1
    
    # Convert to DataFrame
    skills_by_title_df = pd.DataFrame.from_dict(
        {title: dict(skills) for title, skills in title_skills.items()}, 
        orient='index'
    ).fillna(0)
    
    # Add total jobs count for each title
    title_counts = df[title_column].value_counts()
    skills_by_title_df['total_jobs'] = skills_by_title_df.index.map(title_counts)
    
    # Reorder columns to put total_jobs first
    cols = ['total_jobs'] + [col for col in skills_by_title_df.columns if col != 'total_jobs']
    
    return skills_by_title_df[cols].astype(int)

def get_top_skills_by_title(df: pd.DataFrame, 
                          title_column: str = 'title_category',
                          skills_column: str = 'skills',
                          n_skills: int = 5) -> Dict[str, List[str]]:
    """
    Gets the top N skills for each job title.
    
    Parameters:
    df (DataFrame): DataFrame containing title and skills data
    title_column (str): Name of the column containing job titles
    skills_column (str): Name of the column containing skills
    n_skills (int): Number of top skills to return for each title
    
    Returns:
    Dict: Dictionary with titles as keys and lists of top skills as values
    """
    skills_by_title = analyze_skills_by_title(df, title_column, skills_column)
    top_skills = {}
    
    for title in skills_by_title.index:
        title_data = skills_by_title.loc[title].drop('total_jobs')
        top_skills[title] = title_data.nlargest(n_skills).index.tolist()
    
    return top_skills

def get_title_statistics(df: pd.DataFrame,
                       title_column: str = 'title_category',
                       skills_column: str = 'skills') -> pd.DataFrame:
    """
    Get statistics for each title category.
    
    Parameters:
    df (DataFrame): DataFrame containing title and skills data
    title_column (str): Name of the column containing job titles
    skills_column (str): Name of the column containing skills
    
    Returns:
    DataFrame: Statistics for each title category
    """
    stats = []
    
    for title in df[title_column].unique():
        title_df = df[df[title_column] == title]
        skills_lists = title_df[skills_column].apply(get_skill_list)
        
        # Calculate statistics
        skill_lengths = skills_lists.apply(len)
        avg_skills = skill_lengths.mean()
        total_unique_skills = len(set(skill for skills in skills_lists for skill in skills if skill))
        
        stats.append({
            'title': title,
            'job_count': len(title_df),
            'avg_skills_per_job': round(avg_skills, 1),
            'unique_skills_count': total_unique_skills,
            'min_skills': skill_lengths.min(),
            'max_skills': skill_lengths.max()
        })
    
    return pd.DataFrame(stats).sort_values('job_count', ascending=False)

def prepare_skill_matrix_for_visualization(df: pd.DataFrame,
                                         title_column: str = 'title_category',
                                         skills_column: str = 'skills',
                                         min_jobs_per_title: int = 3,
                                         top_n_skills: int = 10) -> pd.DataFrame:
    """
    Prepare normalized skill matrix for visualization.
    
    Parameters:
    df (DataFrame): Input DataFrame
    title_column (str): Column name for job titles
    skills_column (str): Column name for skills
    min_jobs_per_title (int): Minimum jobs required to include a title
    top_n_skills (int): Number of top skills to include
    
    Returns:
    DataFrame: Normalized skill matrix ready for visualization
    """
    # Get skills by title
    skills_by_title = analyze_skills_by_title(df, title_column, skills_column)
    
    # Filter titles with sufficient data
    valid_titles = skills_by_title[skills_by_title['total_jobs'] >= min_jobs_per_title].index
    filtered_data = skills_by_title.loc[valid_titles]
    
    # Get top skills across all titles
    skill_sums = filtered_data.drop('total_jobs', axis=1).sum()
    top_skills = skill_sums.nlargest(top_n_skills).index.tolist()
    
    # Normalize by number of jobs per title
    skill_data = filtered_data[top_skills]
    normalized_matrix = skill_data.div(filtered_data['total_jobs'], axis=0)
    
    return normalized_matrix