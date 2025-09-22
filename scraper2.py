import requests
import pandas as pd
import time
from typing import List, Dict

class JobAPIClient:
    def __init__(self, api_key: str, app_id: str):
        self.api_key = api_key
        self.app_id = app_id
        self.base_url = "https://api.adzuna.com/v1/api"
        self.session = requests.Session()
        
    def search_data_science_jobs(self, country: str = 'us', max_results: int = 50) -> List[Dict]:
        """Search for data science jobs on Adzuna."""
        url = f"{self.base_url}/jobs/{country}/search/1"
        
        params = {
            'app_id': self.app_id,
            'app_key': self.api_key,
            'results_per_page': min(max_results, 50),
            'what': 'data scientist OR machine learning engineer OR data analyst',
            'what_or': 'data science OR ml OR ai engineer',
            'content-type': 'application/json'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            jobs = []
            for job in data.get('results', []):
                jobs.append({
                    'title': job.get('title'),
                    'company': job.get('company', {}).get('display_name'),
                    'location': job.get('location', {}).get('display_name'),
                    'description': job.get('description'),
                    'salary_min': job.get('salary_min'),
                    'salary_max': job.get('salary_max'),
                    'salary_currency': job.get('salary_currency'),
                    'contract_type': job.get('contract_type'),
                    'created': job.get('created'),
                    'url': job.get('redirect_url'),
                    'category': job.get('category', {}).get('label'),
                    'source': 'Adzuna'
                })
            
            return jobs
            
        except Exception as e:
            print(f"Error fetching jobs from Adzuna: {e}")
            return []
    
    def search_by_skills(self, skills: List[str], country: str = 'us', max_results: int = 30) -> List[Dict]:
        """Search jobs requiring specific skills."""
        skills_query = ' OR '.join(skills)
        url = f"{self.base_url}/jobs/{country}/search/1"
        
        params = {
            'app_id': self.app_id,
            'app_key': self.api_key,
            'results_per_page': min(max_results, 50),
            'what': skills_query,
            'content-type': 'application/json'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data.get('results', [])
            
        except Exception as e:
            print(f"Error searching skills on Adzuna: {e}")
            return []

# Example usage
def main():
    # Get free API key from https://developer.adzuna.com/
    API_KEY = "def9006fa9cc19f0139ab75063b29abb"
    APP_ID = "25a97320"
    
    client = JobAPIClient(API_KEY, APP_ID)
    
    # Search for data science jobs
    jobs = client.search_data_science_jobs(country='us', max_results=20)
    
    if jobs:
        df = pd.DataFrame(jobs)
        print(f"Found {len(jobs)} data science jobs")
        print(df[['title', 'company', 'location', 'salary_min', 'salary_max']].head())
        
        # Save to CSV
        df.to_csv('data/raw/adzuna_jobs.csv', index=False)
        print("Jobs saved to adzuna_jobs.csv")
if __name__ == "__main__":
    main()