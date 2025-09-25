# src/processing/clean_skills.py

import pandas as pd
import numpy as np
from datetime import datetime
import ast
import re
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataScienceJobsCleaner:
    """
    Clean and preprocess data science job listings from Found.dev API
    """
    
    # Common data science skill categories for classification
    SKILL_CATEGORIES = {
        'programming': ['python', 'r', 'sql', 'java', 'scala', 'c++', 'javascript', 'julia'],
        'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'mxnet', 'caffe'],
        'big_data': ['spark', 'hadoop', 'hive', 'kafka', 'airflow', 'dbt', 'snowflake'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
        'visualization': ['tableau', 'powerbi', 'matplotlib', 'seaborn', 'plotly', 'd3'],
        'statistics': ['statistics', 'hypothesis testing', 'experimentation', 'a/b testing'],
        'ml_techniques': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'reinforcement learning']
    }
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the cleaner with data directory paths
        
        Args:
            data_dir: Root data directory path
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, batch_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load raw data from batch CSV files
        
        Args:
            batch_files: Specific batch files to load. If None, loads all batches
            
        Returns:
            Combined DataFrame of all raw data
        """
        if batch_files is None:
            batch_files = list(self.raw_dir.glob("jobs_batch_*.csv"))
        
        if not batch_files:
            raise FileNotFoundError("No batch files found in raw directory")
        
        data_frames = []
        for batch_file in batch_files:
            logger.info(f"Loading {batch_file.name}")
            df = pd.read_csv(batch_file)
            df['batch_source'] = batch_file.name
            data_frames.append(df)
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} total records from {len(batch_files)} batches")
        
        return combined_df
    
    def clean_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and enhance location information
        
        Args:
            df: Raw DataFrame with location columns
            
        Returns:
            DataFrame with cleaned location data
        """
        df_clean = df.copy()
        
        # Fix the location string when city/country are NaN
        df_clean['location'] = df_clean['location'].replace(',', 'Remote/Unknown')
                
        # Extract country if available in location string
        def extract_country(location):
            if pd.isna(location):
                return 'Unknown'
            location_str = str(location).lower()
            country = location_str.split(',')[-1].strip().split(':')[0].title()
            return country if country else 'Unknown'
        
        df_clean['country_cleaned'] = df_clean['location'].apply(extract_country)
        
        logger.info("✅ Location data cleaned")
        return df_clean
    
    def parse_skills(self, skills_str: str) -> List[str]:
        """
        Parse skills string into list of individual skills
        
        Args:
            skills_str: Comma-separated skills string
            
        Returns:
            List of cleaned skill names
        """
        if pd.isna(skills_str):
            return []
        
        try:
            # Handle string representation of list
            if skills_str.startswith('[') and skills_str.endswith(']'):
                skills_list = ast.literal_eval(skills_str)
            else:
                # Split by comma and clean
                skills_list = [skill.strip() for skill in skills_str.split(',')]
            
            # Clean each skill name
            cleaned_skills = []
            for skill in skills_list:
                if skill:  # Skip empty strings
                    # Remove extra whitespace and standardize case
                    cleaned_skill = skill.strip().title()
                    cleaned_skills.append(cleaned_skill)
            
            return list(set(cleaned_skills))  # Remove duplicates
            
        except (ValueError, SyntaxError):
            logger.warning(f"Could not parse skills string: {skills_str}")
            return []
    
    def categorize_skills(self, skills_list: List[str]) -> Dict[str, List[str]]:
        """
        Categorize skills into predefined categories
        
        Args:
            skills_list: List of skill names
            
        Returns:
            Dictionary mapping categories to skills
        """
        categorized = {category: [] for category in self.SKILL_CATEGORIES.keys()}
        
        for skill in skills_list:
            skill_lower = skill.lower()
            categorized_flag = False
            
            for category, keywords in self.SKILL_CATEGORIES.items():
                if any(keyword in skill_lower for keyword in keywords):
                    categorized[category].append(skill)
                    categorized_flag = True
            
            if not categorized_flag:
                if 'other' not in categorized:
                    categorized['other'] = []
                categorized['other'].append(skill)
        
        return categorized
    
    def enhance_skills_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse and enhance skills information
        
        Args:
            df: DataFrame with skills column
            
        Returns:
            DataFrame with enhanced skills data
        """
        df_clean = df.copy()
        
        # Parse skills string into list
        df_clean['skills_parsed'] = df_clean['skills'].apply(self.parse_skills)
        
        # Count number of skills
        df_clean['skills_count'] = df_clean['skills_parsed'].apply(len)
        
        # Categorize skills
        df_clean['skills_categorized'] = df_clean['skills_parsed'].apply(self.categorize_skills)
        
        # Create binary columns for major skill categories
        for category in self.SKILL_CATEGORIES.keys():
            df_clean[f'has_{category}'] = df_clean['skills_categorized'].apply(
                lambda x: len(x.get(category, [])) > 0
            )
        
        # Extract primary skill (first skill in list)
        df_clean['primary_skill'] = df_clean['skills_parsed'].apply(
            lambda x: x[0] if x else 'Unknown'
        )
        
        logger.info("✅ Skills data enhanced")
        return df_clean
    
    def clean_job_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and categorize job types
        
        Args:
            df: DataFrame with type column
            
        Returns:
            DataFrame with cleaned job types
        """
        df_clean = df.copy()
        
        # Standardize job type names
        type_mapping = {
            'full_time': ['full-time', 'full time', 'permanent'],
            'part_time': ['part-time', 'part time'],
            'contract': ['contract', 'freelance', 'temporary'],
            'internship': ['internship', 'intern']
        }
        
        def standardize_job_type(job_type):
            if pd.isna(job_type):
                return 'Unknown'
            
            job_type_lower = str(job_type).lower()
            
            for standardized, variants in type_mapping.items():
                if any(variant in job_type_lower for variant in variants):
                    return standardized
            
            return job_type_lower  # Return original if no match
        
        df_clean['type_cleaned'] = df_clean['type'].apply(standardize_job_type)
        
        logger.info("✅ Job types cleaned")
        return df_clean
    
    def clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and enhance salary information
        
        Args:
            df: DataFrame with salary columns
            
        Returns:
            DataFrame with cleaned salary data
        """
        df_clean = df.copy()
        
        # Convert 0 values to NaN (likely missing data)
        salary_cols = ['salary', 'salary_min', 'salary_max']
        for col in salary_cols:
            df_clean[col] = df_clean[col].replace(0, np.nan)
        
        def clean_salary_data(salary_series):
            """
            Clean salary data and extract min/max salaries converted to USD annually
            """
            # Currency conversion rates (approximate rates - you may want to update these)
            EXCHANGE_RATES = {
                'PLN': 0.25,    # Polish Zloty to USD
                'EUR': 1.10,    # Euro to USD
                'INR': 0.012,   # Indian Rupee to USD
                'CAD': 0.75,    # Canadian Dollar to USD
                'GBP': 1.27,    # British Pound to USD
                '£': 1.27,      # British Pound symbol
                '€': 1.10,      # Euro symbol
                '$': 1.00,      # USD
                'Swiss Francs': 1.12,  # Swiss Franc to USD
                'CHF': 1.12,    # Swiss Franc abbreviation
                '': 1.00        # Default to USD if no currency specified
            }
            
            # Work hour assumptions
            HOURS_PER_YEAR = 2080  # 40 hours/week * 52 weeks
            MONTHS_PER_YEAR = 12
            
            def extract_and_convert_salary(salary_str):
                try:
                    if pd.isna(salary_str) or not isinstance(salary_str, (str, float, int)):
                        return np.nan, np.nan
                        
                    salary_str = str(salary_str).strip()
                    
                    if salary_str in ['', 'Market rate', 'Competitive, as per company policy', 'Negotiable']:
                        return np.nan, np.nan
                    
                    # Skip non-salary entries
                    skip_terms = ['Market rate', 'Competitive', 'Negotiable']
                    if any(term in salary_str for term in skip_terms):
                        return np.nan, np.nan
                    
                    # Extract currency
                    currency = 'USD'
                    for curr_symbol, rate in EXCHANGE_RATES.items():
                        if curr_symbol and curr_symbol in salary_str:
                            currency = curr_symbol
                            break
                    
                    # Multiple number extraction strategies
                    numbers = re.findall(r'[\d,]+\.?\d*', salary_str)
                    
                    cleaned_numbers = []
                    for num in numbers:
                        # Remove ALL non-numeric characters except dot and comma
                        cleaned_num = re.sub(r'[^\d,.]', '', str(num))
                        # Remove commas for conversion
                        cleaned_num = cleaned_num.replace(',', '')
                        
                        if cleaned_num and cleaned_num != '.':  # Skip empty strings and lone dots
                            try:
                                num_float = float(cleaned_num)
                                cleaned_numbers.append(num_float)
                            except ValueError:
                                continue
                    
                    if not cleaned_numbers:
                        return np.nan, np.nan
                    
                    # Use min/max of found numbers
                    min_salary = min(cleaned_numbers)
                    max_salary = max(cleaned_numbers)
                    
                    # Handle single number case (use same value for min and max)
                    if len(cleaned_numbers) == 1:
                        max_salary = min_salary
                    
                    # Period conversion
                    salary_lower = salary_str.lower()
                    if 'hourly' in salary_lower or '/h' in salary_lower:
                        min_salary *= HOURS_PER_YEAR
                        max_salary *= HOURS_PER_YEAR
                    elif 'monthly' in salary_lower:
                        min_salary *= MONTHS_PER_YEAR
                        max_salary *= MONTHS_PER_YEAR
                    
                    # Currency conversion
                    exchange_rate = EXCHANGE_RATES.get(currency, 1.0)
                    min_salary_usd = min_salary * exchange_rate
                    max_salary_usd = max_salary * exchange_rate
                    
                    return min_salary_usd, max_salary_usd
                    
                except Exception as e:
                    # Log the error for debugging
                    print(f"Error processing salary: '{salary_str}' - {str(e)}")
                    return np.nan, np.nan
            
            # Apply the function to the series
            results = salary_series.apply(extract_and_convert_salary)
            
            # Create separate columns for min and max salary
            min_salaries = results.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            max_salaries = results.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
            
            return min_salaries, max_salaries

        df_clean['salary_min'], df_clean['salary_max'] = clean_salary_data(df_clean['salary'])
        
        # Calculate salary range where possible
        def calculate_salary_range(row):
            if pd.notna(row['salary_min']) and pd.notna(row['salary_max']):
                return row['salary_max'] - row['salary_min']
            return np.nan
        
        df_clean['salary_range'] = df_clean.apply(calculate_salary_range, axis=1)
        
        # Create salary category
        def categorize_salary(salary):
            salary = int(salary) if pd.notna(salary) else np.nan
            if pd.isna(salary):
                return 'Unknown'
            elif salary < 50000:
                return 'Low (<50k)'
            elif salary < 100000:
                return 'Medium (50k-100k)'
            elif salary < 150000:
                return 'High (100k-150k)'
            else:
                return 'Very High (>150k)'
        
        df_clean['salary_category'] = df_clean['salary_min'].apply(categorize_salary)
        
        logger.info("✅ Salary data cleaned")
        return df_clean
    
    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to proper datetime format
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with converted dates
        """
        df_clean = df.copy()
        
        # Convert published date
        df_clean['published_dt'] = pd.to_datetime(df_clean['published'], utc=True, errors='coerce')
        
        # Extract date components
        df_clean['published_year'] = df_clean['published_dt'].dt.year
        df_clean['published_month'] = df_clean['published_dt'].dt.month
        df_clean['published_week'] = df_clean['published_dt'].dt.isocalendar().week
        
        # Calculate days since publication
        current_date = pd.Timestamp.now(tz='UTC')
        df_clean['days_since_publication'] = (current_date - df_clean['published_dt']).dt.days
        
        logger.info("✅ Date data converted")
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate job listings
        
        Args:
            df: DataFrame to deduplicate
            
        Returns:
            Deduplicated DataFrame
        """
        # Create a unique identifier based on title, company, and published date
        df_clean = df.copy()
        
        # Before deduplication
        initial_count = len(df_clean)
        
        # Remove exact duplicates
        df_clean = df_clean.drop_duplicates(subset=['title', 'company', 'published_dt', 'location','type', 'salary'])
        
        # Remove near-duplicates based on title and company
        df_clean = df_clean.drop_duplicates(
            subset=['title', 'company', 'published_dt'], 
            keep='first'
        )
        
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        logger.info(f"✅ Removed {duplicates_removed} duplicate records")
        
        return df_clean
    
    def clean_title_data(self, df):
        """
        Categorize job titles into standardized roles and seniority levels
        """
        import re
        from collections import Counter
        
        class TitleCategorizer:
            def __init__(self):
                self.title_patterns = {
                    'Data Scientist': [
                        r'data scientist', r'scientist.*data', r'data.*scientist',
                        r'\bds\b', r'data science'
                    ],
                    'Machine Learning Engineer': [
                        r'machine learning engineer', r'ml engineer',
                        r'engineer.*machine learning', r'machine learning.*engineer',
                        r'\bmle\b', r'ai engineer', r'engineer.*ai'
                    ],
                    'Data Analyst': [
                        r'data analyst', r'analyst.*data', r'business analyst',
                        r'business intelligence', r'\bbi\b', r'reporting analyst'
                    ],
                    'Data Engineer': [
                        r'data engineer', r'engineer.*data', r'etl',
                        r'data pipeline', r'big data engineer'
                    ],
                    'MLOps Engineer': [
                        r'mlops', r'machine learning operations',
                        r'ai infrastructure', r'model deployment'
                    ],
                    'AI Engineer': [
                        r'ai engineer', r'artificial intelligence engineer',
                        r'deep learning engineer', r'neural network engineer'
                    ],
                    'Research Scientist': [
                        r'research scientist', r'ai researcher',
                        r'machine learning researcher', r'applied scientist'
                    ],
                    'Data Science Manager': [
                        r'manager.*data', r'data.*manager',
                        r'lead.*data', r'principal.*data',
                        r'head.*data', r'director.*data'
                    ],
                    'Business Analyst': [
                        r'business analyst', r'business intelligence analyst',
                        r'product analyst', r'marketing analyst'
                    ],
                    'Data Architect': [
                        r'data architect', r'solution architect.*data',
                        r'database architect', r'cloud data architect'
                    ],
                    'BI Developer': [
                        r'bi developer', r'business intelligence developer',
                        r'tableau developer', r'power bi', r'qlik'
                    ],
                    'Statistician': [
                        r'statistician', r'statistical analyst',
                        r'quantitative analyst', r'\bquant\b'
                    ]
                }
                
                self.seniority_keywords = {
                    'Junior': [r'junior', r'entry', r'graduate', r'trainee', r'associate', r'level i'],
                    'Senior': [r'senior', r'sr\.', r'lead', r'principal', r'staff', r'experienced'],
                    'Manager': [r'manager', r'director', r'head of', r'chief', r'vp', r'vice president'],
                    'Intern': [r'intern', r'internship', r'student']
                }
            
            def categorize_title(self, title):
                if pd.isna(title) or not isinstance(title, str):
                    return 'Other', 'Unknown'
                
                title_lower = title.lower().strip()
                
                # Determine role category
                role_category = 'Other'
                for category, patterns in self.title_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, title_lower, re.IGNORECASE):
                            role_category = category
                            break
                    if role_category != 'Other':
                        break
                
                # Determine seniority level
                seniority = 'Mid-Level'  # Default
                for level, keywords in self.seniority_keywords.items():
                    for keyword in keywords:
                        if re.search(keyword, title_lower, re.IGNORECASE):
                            seniority = level
                            break
                    if seniority != 'Mid-Level':
                        break
                
                return role_category, seniority
        
        # Apply categorization to the dataframe
        df_clean = df.copy()
        categorizer = TitleCategorizer()
        
        if 'title' in df_clean.columns:
            # Categorize each title
            results = df_clean['title'].apply(categorizer.categorize_title)
            df_clean['role_category'] = results.apply(lambda x: x[0])
            df_clean['seniority_level'] = results.apply(lambda x: x[1])
            
            # Log the distribution
            role_counts = df_clean['role_category'].value_counts()
            logger.info("Title categorization completed:")
            for role, count in role_counts.items():
                logger.info(f"  {role}: {count} jobs")
        
        return df_clean
    
    def run_full_cleaning_pipeline(self, batch_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline
        
        Args:
            batch_files: Specific batch files to process
            
        Returns:
            Fully cleaned DataFrame
        """
        logger.info("🚀 Starting data cleaning pipeline...")
        
        # Load raw data
        df = self.load_raw_data(batch_files)
        
        # Apply cleaning steps
        df_clean = (df
                   .pipe(self.clean_location_data)
                   .pipe(self.enhance_skills_data)
                   .pipe(self.clean_job_type)
                   .pipe(self.clean_salary_data)
                   .pipe(self.convert_dates)
                   .pipe(self.remove_duplicates)
                   .pipe(self.clean_title_data)
                   )
        
        # Final data quality check
        self._quality_check(df_clean)
        
        logger.info(f"🎉 Cleaning complete! Final dataset: {len(df_clean)} records")
        
        return df_clean
    
    def _quality_check(self, df: pd.DataFrame):
        """
        Perform basic data quality checks
        
        Args:
            df: Cleaned DataFrame to check
        """
        logger.info("🔍 Performing data quality checks...")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        total_records = len(df)
        
        for column, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / total_records) * 100
                logger.info(f"   {column}: {missing_count} missing ({percentage:.1f}%)")
        
        # Check data types
        logger.info("📊 Data types:")
        for column, dtype in df.dtypes.items():
            logger.info(f"   {column}: {dtype}")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_type: str = "interim"):
        """
        Save cleaned data to appropriate directory
        
        Args:
            df: Cleaned DataFrame to save
            output_type: "interim" for partially cleaned, "processed" for final
        """
        if output_type == "interim":
            output_path = self.interim_dir / "cleaned_jobs.csv"
        elif output_type == "processed":
            output_path = self.processed_dir / "processed_jobs.csv"
        else:
            raise ValueError("output_type must be 'interim' or 'processed'")
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved {len(df)} records to {output_path}")


# Convenience function for quick usage
def clean_jobs_data(data_dir: str = "../data", save_interim: bool = True) -> pd.DataFrame:
    """
    Convenience function to run the full cleaning pipeline
    
    Args:
        data_dir: Root data directory
        save_interim: Whether to save the cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataScienceJobsCleaner(data_dir)
    cleaned_df = cleaner.run_full_cleaning_pipeline()
    
    if save_interim:
        cleaner.save_cleaned_data(cleaned_df, "interim")
    
    return cleaned_df


if __name__ == "__main__":
    # Example usage when run as script
    df_cleaned = clean_jobs_data()
    print(f"Cleaning complete! Final dataset shape: {df_cleaned.shape}")