"""
Advanced correlation and association analysis functions for job skills.
"""

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from typing import Tuple, Dict, Any
from collections import defaultdict

def create_skill_cooccurrence_matrix(df: pd.DataFrame, 
                                   skills_column: str = 'skills') -> pd.DataFrame:
    """
    Creates a matrix showing how often skills appear together in job postings.
    
    Parameters:
    df (DataFrame): DataFrame containing skills data
    skills_column (str): Name of the column containing skills
    
    Returns:
    DataFrame: Symmetric co-occurrence matrix with skills as rows and columns
    """
    # Get all unique skills
    all_skills = set()
    for skills_list in df[skills_column]:
        if isinstance(skills_list, (list, str)):
            processed_skills = skills_list if isinstance(skills_list, list) else \
                              skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
            all_skills.update([s for s in processed_skills if s])
    
    all_skills = sorted(list(all_skills))
    
    # Initialize co-occurrence matrix with zeros
    cooccurrence_matrix = pd.DataFrame(0, index=all_skills, columns=all_skills)
    
    # Fill the co-occurrence matrix
    for skills_list in df[skills_column]:
        if isinstance(skills_list, (list, str)):
            processed_skills = skills_list if isinstance(skills_list, list) else \
                              skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
            valid_skills = [s for s in processed_skills if s and s in all_skills]
            
            # Count co-occurrences for each pair
            for i in range(len(valid_skills)):
                for j in range(i + 1, len(valid_skills)):
                    skill1, skill2 = valid_skills[i], valid_skills[j]
                    cooccurrence_matrix.loc[skill1, skill2] += 1
                    cooccurrence_matrix.loc[skill2, skill1] += 1  # Matrix is symmetric
    
    return cooccurrence_matrix

def calculate_statistical_correlations(df: pd.DataFrame,
                                     skills_column: str = 'skills') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates point-biserial correlations between all skill pairs.
    
    Parameters:
    df (DataFrame): DataFrame containing skills data
    skills_column (str): Name of the column containing skills
    
    Returns:
    Tuple: (correlation_matrix, pvalue_matrix) - DataFrames with correlation coefficients and p-values
    """
    # Get all unique skills
    all_skills = set()
    for skills_list in df[skills_column]:
        if isinstance(skills_list, (list, str)):
            processed_skills = skills_list if isinstance(skills_list, list) else \
                              skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
            all_skills.update([s for s in processed_skills if s])
    
    all_skills = sorted(list(all_skills))
    
    # Create binary matrix: 1 if skill is present, 0 if absent
    binary_matrix = []
    for skills_list in df[skills_column]:
        if isinstance(skills_list, (list, str)):
            processed_skills = skills_list if isinstance(skills_list, list) else \
                              skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
            binary_row = [1 if skill in processed_skills else 0 for skill in all_skills]
            binary_matrix.append(binary_row)
        else:
            binary_matrix.append([0] * len(all_skills))
    
    binary_df = pd.DataFrame(binary_matrix, columns=all_skills)
    
    # Calculate correlations for each skill pair
    correlation_results = {}
    for skill1 in all_skills:
        correlation_results[skill1] = {}
        for skill2 in all_skills:
            if skill1 == skill2:
                # Self-correlation is always 1
                correlation_results[skill1][skill2] = {'correlation': 1.0, 'p_value': 0.0}
            else:
                try:
                    corr, p_value = pointbiserialr(binary_df[skill1], binary_df[skill2])
                    correlation_results[skill1][skill2] = {'correlation': corr, 'p_value': p_value}
                except:
                    # Handle cases where calculation fails (e.g., constant values)
                    correlation_results[skill1][skill2] = {'correlation': 0.0, 'p_value': 1.0}
    
    # Create separate DataFrames for correlations and p-values
    corr_matrix = pd.DataFrame({skill: {s: results['correlation'] for s, results in inner_dict.items()} 
                               for skill, inner_dict in correlation_results.items()})
    pvalue_matrix = pd.DataFrame({skill: {s: results['p_value'] for s, results in inner_dict.items()} 
                                 for skill, inner_dict in correlation_results.items()})
    
    return corr_matrix, pvalue_matrix

def analyze_skill_associations(df: pd.DataFrame,
                             target_skill: str = 'python',
                             skills_column: str = 'skills') -> pd.DataFrame:
    """
    Comprehensive analysis of associations for a target skill.
    
    Parameters:
    df (DataFrame): DataFrame containing skills data
    target_skill (str): The skill to analyze associations for
    skills_column (str): Name of the column containing skills
    
    Returns:
    DataFrame: Association metrics for skills related to the target skill
    """
    # Create co-occurrence matrix
    cooccurrence_matrix = create_skill_cooccurrence_matrix(df, skills_column)
    
    # Check if target skill exists in the data
    if target_skill not in cooccurrence_matrix.index:
        print(f"Warning: Target skill '{target_skill}' not found in data")
        return pd.DataFrame()
    
    # Get co-occurrence counts with target skill
    target_cooccurrence = cooccurrence_matrix[target_skill].sort_values(ascending=False)
    target_cooccurrence = target_cooccurrence.drop(target_skill)  # Remove self
    
    total_jobs = len(df)
    
    # Count jobs with target skill
    target_jobs = 0
    for skills_list in df[skills_column]:
        if isinstance(skills_list, (list, str)):
            processed_skills = skills_list if isinstance(skills_list, list) else \
                              skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
            if target_skill in processed_skills:
                target_jobs += 1
    
    # Calculate association metrics
    results = []
    for skill, cooccur_count in target_cooccurrence.items():
        # Count jobs with this skill
        skill_jobs = 0
        for skills_list in df[skills_column]:
            if isinstance(skills_list, (list, str)):
                processed_skills = skills_list if isinstance(skills_list, list) else \
                                  skills_list[1:-1].replace(' ', '').replace("'", '').split(',')
                if skill in processed_skills:
                    skill_jobs += 1
        
        # Calculate association metrics
        association_ratio = cooccur_count / target_jobs if target_jobs > 0 else 0
        lift = (cooccur_count / total_jobs) / ((target_jobs / total_jobs) * (skill_jobs / total_jobs)) \
               if target_jobs > 0 and skill_jobs > 0 and total_jobs > 0 else 0
        
        results.append({
            'skill': skill,
            'cooccurrence_count': cooccur_count,
            'association_ratio': round(association_ratio, 3),
            'lift_score': round(lift, 3),
            'skill_frequency': skill_jobs,
            'target_skill_frequency': target_jobs,
            'cooccurrence_percentage': (cooccur_count / target_jobs * 100) if target_jobs > 0 else 0
        })
    
    return pd.DataFrame(results).sort_values('cooccurrence_count', ascending=False)

def get_significant_correlations(corr_matrix: pd.DataFrame,
                               pvalue_matrix: pd.DataFrame,
                               correlation_threshold: float = 0.3,
                               pvalue_threshold: float = 0.05) -> pd.DataFrame:
    """
    Get statistically significant correlations from correlation matrices.
    
    Parameters:
    corr_matrix (DataFrame): Correlation coefficient matrix
    pvalue_matrix (DataFrame): P-value matrix
    correlation_threshold (float): Minimum absolute correlation to consider
    pvalue_threshold (float): Maximum p-value to consider significant
    
    Returns:
    DataFrame: Significant correlations with metrics
    """
    significant_correlations = []
    
    for skill1 in corr_matrix.index:
        for skill2 in corr_matrix.columns:
            if skill1 != skill2:  # Skip self-correlations
                corr = corr_matrix.loc[skill1, skill2]
                p_value = pvalue_matrix.loc[skill1, skill2]
                
                if (abs(corr) >= correlation_threshold) and (p_value <= pvalue_threshold):
                    significant_correlations.append({
                        'skill1': skill1,
                        'skill2': skill2,
                        'correlation': corr,
                        'p_value': p_value,
                        'abs_correlation': abs(corr)
                    })
    
    return pd.DataFrame(significant_correlations).sort_values('abs_correlation', ascending=False)