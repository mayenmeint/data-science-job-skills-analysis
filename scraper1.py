import requests
import pandas as pd
from typing import List, Dict, Optional
import time
class USAJobsClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://data.usajobs.gov/api/search"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization-Key': api_key,
            'User-Agent': 'YourApp/1.0 (your@email.com)'
        })
    
    def search_data_jobs(self, keywords: List[str] = None, max_results: int = 1000) -> List[Dict]:
        """Search for data-related government jobs."""
        if keywords is None:
            keywords = ['data scientist', 'data analyst', 'statistician', 'computer scientist']
        
        jobs = []
        
        for keyword in keywords:
            params = {
                'Keyword': keyword,
                'ResultsPerPage': min(max_results, 1000),
                'Page': 1
            }
            
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                for job in data.get('SearchResult', {}).get('SearchResultItems', []):
                    job_data = job.get('MatchedObjectDescriptor', {})
                    jobs.append({
                        'title': job_data.get('PositionTitle'),
                        'department': job_data.get('DepartmentName'),
                        'agency': job_data.get('OrganizationName'),
                        'location': job_data.get('PositionLocationDisplay'),
                        'salary_min': job_data.get('PositionRemuneration', [{}])[0].get('MinimumRange'),
                        'salary_max': job_data.get('PositionRemuneration', [{}])[0].get('MaximumRange'),
                        'url': job_data.get('PositionURI'),
                        'date_posted': job_data.get('PublicationStartDate'),
                        'description': job_data.get('UserArea', {}).get('Details', {}).get('JobSummary', ''),
                        'source': 'USAJobs'
                    })
                
                time.sleep(1)  # Respect rate limits
                
            except Exception as e:
                print(f"Error searching for {keyword}: {e}")
                continue
        
        return jobs

# Usage
def fetch_government_jobs():
    # Get free API key from https://developer.usajobs.gov/
    API_KEY = "RGWBrkbCrA+WQr1HpSHfxVHnw6P2Intr4slXFsb8mKQ="
    
    client = USAJobsClient(API_KEY)
    jobs = client.search_data_jobs(max_results=50)
    
    if jobs:
        df = pd.DataFrame(jobs)
        df.to_csv('data/raw/usajobs_data.csv', index=False)
        print(f"Saved {len(jobs)} government jobs")
    return jobs

fetch_government_jobs()