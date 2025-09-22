import requests
import pandas as pd
import time
import json
from datetime import datetime
from typing import List, Dict, Optional

class DataScienceJobsScraper:
    def __init__(self):
        self.base_url = "https://jobs.github.com/positions"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        
        # Data science related search terms
        self.data_science_keywords = [
            "data scientist", "data science", "machine learning", 
            "ml engineer", "ai engineer", "data analyst",
            "business intelligence", "data engineer", "data mining",
            "statistical analyst", "predictive modeling", "deep learning",
            "nlp", "natural language processing", "computer vision",
            "big data", "data visualization", "quantitative analyst"
        ]
    
    def scrape_all_data_science_jobs(self, pages: int = 10) -> List[Dict]:
        """
        Scrape all data science related jobs from any location worldwide
        """
        all_jobs = []
        
        for page in range(1, pages + 1):
            try:
                print(f"Scraping page {page} for data science jobs...")
                
                response = self.session.get(
                    f"{self.base_url}.json",
                    params={'page': page},
                    headers=self.headers,
                    timeout=15
                )
                response.raise_for_status()
                
                jobs_data = response.json()
                
                if not jobs_data:
                    print(f"No more jobs found on page {page}")
                    break
                
                # Filter for data science related jobs
                data_science_jobs = self._filter_data_science_jobs(jobs_data)
                
                if data_science_jobs:
                    all_jobs.extend(data_science_jobs)
                    print(f"Page {page}: Found {len(data_science_jobs)} data science jobs")
                else:
                    print(f"Page {page}: No data science jobs found")
                
                time.sleep(1.5)  # Respectful delay
                
            except requests.exceptions.RequestException as e:
                print(f"Error scraping page {page}: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from page {page}: {e}")
                break
        
        return all_jobs
    
    def _filter_data_science_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """
        Filter jobs to only include data science related positions
        """
        data_science_jobs = []
        
        for job in jobs:
            if self._is_data_science_job(job):
                processed_job = self._process_job_data(job)
                if processed_job:
                    data_science_jobs.append(processed_job)
        
        return data_science_jobs
    
    def _is_data_science_job(self, job: Dict) -> bool:
        """
        Check if a job is related to data science
        """
        title = job.get('title', '').lower()
        description = job.get('description', '').lower()
        
        # Check if any data science keyword is in title or description
        for keyword in self.data_science_keywords:
            if keyword in title or keyword in description:
                return True
        
        return False
    
    def _process_job_data(self, job_data: Dict) -> Dict:
        """
        Process and enrich job data
        """
        try:
            processed_job = {
                'id': job_data.get('id', ''),
                'title': job_data.get('title', 'No Title').strip(),
                'company': job_data.get('company', 'Unknown Company').strip(),
                'company_url': job_data.get('company_url', ''),
                'location': job_data.get('location', 'Remote').strip(),
                'type': job_data.get('type', 'Full Time').strip(),
                'description': job_data.get('description', ''),
                'how_to_apply': job_data.get('how_to_apply', ''),
                'company_logo': job_data.get('company_logo', ''),
                'url': job_data.get('url', ''),
                'created_at': job_data.get('created_at', ''),
                'scraped_at': datetime.now().isoformat(),
                'category': self._categorize_job(job_data)
            }
            
            return processed_job
            
        except Exception as e:
            print(f"Error processing job data: {e}")
            return None
    
    def _categorize_job(self, job: Dict) -> str:
        """
        Categorize the data science job
        """
        title = job.get('title', '').lower()
        description = job.get('description', '').lower()
        
        categories = {
            'data_scientist': ['data scientist', 'data science'],
            'machine_learning': ['machine learning', 'ml engineer', 'ai engineer'],
            'data_analyst': ['data analyst', 'business intelligence'],
            'data_engineer': ['data engineer', 'etl', 'data pipeline'],
            'research_scientist': ['research scientist', 'research engineer'],
            'nlp': ['nlp', 'natural language processing'],
            'computer_vision': ['computer vision', 'cv engineer'],
            'quantitative': ['quantitative', 'quant analyst']
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in title or keyword in description:
                    return category
        
        return 'other_data_science'
    
    def search_specific_data_science_roles(self, specific_role: str, pages: int = 5) -> List[Dict]:
        """
        Search for specific data science roles
        """
        all_jobs = []
        
        for page in range(1, pages + 1):
            try:
                response = self.session.get(
                    f"{self.base_url}.json",
                    params={
                        'description': specific_role,
                        'page': page
                    },
                    headers=self.headers,
                    timeout=15
                )
                response.raise_for_status()
                
                jobs_data = response.json()
                
                if not jobs_data:
                    break
                
                # Filter and process jobs
                filtered_jobs = []
                for job in jobs_data:
                    processed_job = self._process_job_data(job)
                    if processed_job:
                        filtered_jobs.append(processed_job)
                
                all_jobs.extend(filtered_jobs)
                print(f"Page {page}: Found {len(filtered_jobs)} {specific_role} jobs")
                
                time.sleep(1.5)
                
            except Exception as e:
                print(f"Error searching for {specific_role}: {e}")
                break
        
        return all_jobs
    
    def analyze_data_science_jobs(self, jobs: List[Dict]) -> Dict:
        """
        Analyze the data science job market
        """
        if not jobs:
            return {}
        
        df = pd.DataFrame(jobs)
        
        analysis = {
            'total_jobs': len(jobs),
            'unique_companies': df['company'].nunique(),
            'job_categories': df['category'].value_counts().to_dict(),
            'top_locations': df['location'].value_counts().head(15).to_dict(),
            'job_types': df['type'].value_counts().to_dict(),
            'top_companies': df['company'].value_counts().head(10).to_dict(),
            'recent_jobs': len([job for job in jobs if self._is_recent(job.get('created_at', ''))])
        }
        
        return analysis
    
    def _is_recent(self, created_at: str) -> bool:
        """
        Check if job was posted recently (within last 7 days)
        """
        try:
            if not created_at:
                return False
            
            # Parse the date string (format: "Wed Oct 25 2023 14:30:00 GMT+0000")
            date_str = ' '.join(created_at.split()[:4])
            job_date = datetime.strptime(date_str, '%a %b %d %Y')
            days_diff = (datetime.now() - job_date).days
            
            return days_diff <= 7
            
        except:
            return False
    
    def save_jobs(self, jobs: List[Dict], filename: str = "data_science_jobs"):
        """
        Save jobs to CSV and JSON files
        """
        if not jobs:
            print("No jobs to save")
            return
        
        try:
            # Save to CSV
            df = pd.DataFrame(jobs)
            csv_filename = f"{filename}.csv"
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"✓ Saved {len(jobs)} jobs to {csv_filename}")
            
            # Save to JSON
            json_filename = f"{filename}.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(jobs)} jobs to {json_filename}")
            
        except Exception as e:
            print(f"Error saving jobs: {e}")
    
    def display_summary(self, jobs: List[Dict]):
        """
        Display summary of found data science jobs
        """
        if not jobs:
            print("No data science jobs found!")
            return
        
        analysis = self.analyze_data_science_jobs(jobs)
        
        print(f"\n{'='*80}")
        print(f"📊 DATA SCIENCE JOBS SUMMARY - WORLDWIDE")
        print(f"{'='*80}")
        print(f"Total jobs found: {analysis['total_jobs']}")
        print(f"Unique companies: {analysis['unique_companies']}")
        print(f"Recent jobs (last 7 days): {analysis['recent_jobs']}")
        
        print(f"\n📍 Top locations:")
        for location, count in list(analysis['top_locations'].items())[:5]:
            print(f"   {location}: {count} jobs")
        
        print(f"\n🏢 Top companies:")
        for company, count in list(analysis['top_companies'].items())[:5]:
            print(f"   {company}: {count} jobs")
        
        print(f"\n📋 Job categories:")
        for category, count in analysis['job_categories'].items():
            print(f"   {category}: {count} jobs")

# Main function to scrape data science jobs
def main():
    # Initialize the scraper
    scraper = DataScienceJobsScraper()
    
    print("🌍 Scraping Data Science Jobs from GitHub Jobs (Worldwide)...")
    print("This may take a few minutes...\n")
    
    # Scrape all data science jobs (10 pages = ~500 jobs)
    data_science_jobs = scraper.scrape_all_data_science_jobs(pages=10)
    
    if data_science_jobs:
        # Display summary
        scraper.display_summary(data_science_jobs)
        
        # Save all jobs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"data_science_jobs_worldwide_{timestamp}"
        scraper.save_jobs(data_science_jobs, filename)
        
        # Show sample of jobs
        print(f"\n{'='*80}")
        print(f"🎯 SAMPLE JOBS (First 5)")
        print(f"{'='*80}")
        
        for i, job in enumerate(data_science_jobs[:5]):
            print(f"\n{i+1}. {job['title']}")
            print(f"   🏢 {job['company']}")
            print(f"   📍 {job['location']}")
            print(f"   📋 {job['category']}")
            print(f"   🔗 {job['url']}")
            print(f"   {'-'*60}")
        
        print(f"\n✅ Scraping complete! Found {len(data_science_jobs)} data science jobs worldwide.")
        
    else:
        print("❌ No data science jobs found. Try increasing the number of pages.")

# Run additional searches for specific roles
def search_specific_roles():
    scraper = DataScienceJobsScraper()
    
    specific_roles = [
        "machine learning engineer",
        "data analyst",
        "nlp engineer",
        "data engineer"
    ]
    
    for role in specific_roles:
        print(f"\n🔍 Searching for '{role}' jobs...")
        jobs = scraper.search_specific_data_science_roles(role, pages=10)
        
        if jobs:
            print(f"Found {len(jobs)} {role} jobs")
            filename = f"{role.replace(' ', '_')}_jobs"
            scraper.save_jobs(jobs, filename)

if __name__ == "__main__":
    # Run main scraper for all data science jobs
    main()
    
    # Uncomment to run specific role searches
    #search_specific_roles()