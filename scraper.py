# job_scraper_manager.py

import os
import time
import random
import json
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# --------------------- CONFIG ---------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # ensure data directory exists

HEADERS = {
    'User-Agent': 'DataScienceJobScraper/5.0 (https://myportfolio.com; contact@email.com)'
}

DEFAULT_MIN_DELAY = 0.18
DEFAULT_MAX_DELAY = 0.45

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

# --------------------- HELPERS ---------------------
def safe_request(session: requests.Session, url: str, max_retries: int = 3, timeout: int = 15) -> Optional[requests.Response]:
    """Session-aware request with retries. Returns Response or None."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1 * (attempt + 1))
    return None

def rate_limited_request(session: requests.Session, url: str, min_delay=DEFAULT_MIN_DELAY, max_delay=DEFAULT_MAX_DELAY, **kwargs):
    """Sleep a small random jitter then call safe_request."""
    time.sleep(random.uniform(min_delay, max_delay))
    return safe_request(session, url, **kwargs)

def parallel_map(fn, inputs: List, max_workers: int = 5):
    """Run fn(item) in parallel, return results (exceptions logged)."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(fn, item): item for item in inputs}
        for future in as_completed(future_to_item):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"worker error: {e}")
    return results

def extract_skills(description_text: str) -> List[str]:
    """Extract known tech skills from job description."""
    if not description_text:
        return []
    text_lower = description_text.lower()
    return [s for s in TECH_SKILLS if re.search(r'\b' + re.escape(s) + r'\b', text_lower)]

# --------------------- SCRAPER CLASS ---------------------
class JobScraperManager:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    # --------------------- REMOTE JOB BOARDS ---------------------
    def scrape_remoteok(self) -> List[Dict]:
        """Fetch RemoteOK jobs via their API/JSON blob."""
        url = "https://remoteok.com/api"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            print("RemoteOK: failed to fetch API")
            return jobs
        try:
            data = resp.json()
            for job in data[1:]:  # skip first metadata element
                desc = BeautifulSoup(job.get("description",""), "html.parser").get_text(separator="\n").strip()
                jobs.append({
                    'company': job.get("company"),
                    'title': job.get("position"),
                    'location': 'Remote',
                    'url': f"https://remoteok.com/remote-jobs/{job.get('slug')}",
                    'description': desc,
                    'skills': ', '.join(extract_skills(desc)),
                    'source': 'remoteok'
                })
        except Exception as e:
            print("RemoteOK parse error:", e)
        self.save_csv(jobs, "REMOTEOK_JOBS.csv")
        return jobs

    def scrape_weworkremotely(self, max_workers=6) -> List[Dict]:
        """Scrape WeWorkRemotely jobs with detail page descriptions."""
        url = "https://weworkremotely.com/categories/remote-data-science-jobs"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            print("WWR: failed to fetch category page")
            return jobs
        soup = BeautifulSoup(resp.text, 'html.parser')
        entries = []
        for li in soup.find_all('li', class_='feature'):
            a = li.find('a', href=True)
            company = li.find('span', class_='company')
            title = li.find('span', class_='title')
            if a and company and title:
                entries.append({'company': company.get_text(strip=True),
                                'title': title.get_text(strip=True),
                                'url': f"https://weworkremotely.com{a['href']}"})
        # Fetch detail pages in parallel
        def fetch_detail(e):
            desc = ''
            r = rate_limited_request(self.session, e['url'])
            if r:
                dsoup = BeautifulSoup(r.text, 'html.parser')
                sel = dsoup.find('div', class_='listing-container')
                if sel:
                    desc = sel.get_text("\n", strip=True)
            e['description'] = desc
            e['skills'] = ', '.join(extract_skills(desc))
            e['location'] = 'Remote'
            e['source'] = 'weworkremotely'
            return e
        jobs = parallel_map(fetch_detail, entries, max_workers=max_workers)
        self.save_csv(jobs, "WWR_JOBS.csv")
        return jobs

    def scrape_remotive(self) -> List[Dict]:
        """Scrape Remotive public API."""
        url = "https://remotive.com/api/remote-jobs"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            print("Remotive: failed to fetch API")
            return jobs
        try:
            data = resp.json()
            for job in data.get('jobs', []):
                desc = job.get('description') or ''
                jobs.append({
                    'company': job.get('company_name'),
                    'title': job.get('title'),
                    'location': job.get('candidate_required_location') or 'Remote',
                    'url': job.get('url'),
                    'description': desc,
                    'skills': ', '.join(extract_skills(desc)),
                    'source': 'remotive'
                })
        except Exception as e:
            print("Remotive parse error:", e)
        self.save_csv(jobs, "REMOTIVE_JOBS.csv")
        return jobs

    def scrape_arbeitnow(self) -> List[Dict]:
        """Scrape ArbeitNow public API."""
        url = "https://www.arbeitnow.com/api/job-board-api"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            print("ArbeitNow: failed to fetch API")
            return jobs
        try:
            data = resp.json()
            for j in data.get('data', []):
                desc = j.get('description', '') or ''
                jobs.append({
                    'company': j.get('company') or j.get('company_name'),
                    'title': j.get('title'),
                    'location': j.get('location') or 'Remote',
                    'url': j.get('url'),
                    'description': desc,
                    'skills': ', '.join(extract_skills(desc)),
                    'source': 'arbeitnow'
                })
        except Exception as e:
            print("ArbeitNow parse error:", e)
        self.save_csv(jobs, "ARBEITNOW_JOBS.csv")
        return jobs

    # --------------------- ENTERPRISE ATS SCRAPERS ---------------------
    def scrape_greenhouse(self, company_key: str, company_name: str, max_workers=5) -> List[Dict]:
        """Scrape Greenhouse company jobs."""
        url = f"https://boards-api.greenhouse.io/v1/boards/{company_key}/jobs"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            return jobs
        try:
            data = resp.json()
            job_list = data.get('jobs', [])
            results = []
            def fetch_detail(job):
                desc = ''
                job_id = job.get('id')
                if job_id:
                    detail_url = f"https://boards-api.greenhouse.io/v1/boards/{company_key}/jobs/{job_id}"
                    r = rate_limited_request(self.session, detail_url)
                    if r:
                        desc = r.json().get('content', '')
                return {
                    'company': company_name,
                    'title': job.get('title'),
                    'location': job.get('location', {}).get('name'),
                    'url': job.get('absolute_url'),
                    'description': desc,
                    'skills': ', '.join(extract_skills(desc)),
                    'source': 'greenhouse'
                }
            results = parallel_map(fetch_detail, job_list, max_workers=max_workers)
            return results
        except Exception as e:
            print(f"Greenhouse parse error ({company_key}):", e)
            return []

    def scrape_lever(self, company_key: str, company_name: str, max_workers=5) -> List[Dict]:
        """Scrape Lever company jobs."""
        url = f"https://api.lever.co/v0/postings/{company_key}"
        jobs = []
        resp = safe_request(self.session, url)
        if not resp:
            return jobs
        try:
            data = resp.json()
            def fetch_detail(job):
                desc = job.get('descriptionPlain') or job.get('description','')
                return {
                    'company': company_name,
                    'title': job.get('text'),
                    'location': job.get('categories', {}).get('location'),
                    'url': job.get('hostedUrl'),
                    'description': desc,
                    'skills': ', '.join(extract_skills(desc)),
                    'source': 'lever'
                }
            jobs = parallel_map(fetch_detail, data, max_workers=max_workers)
            return jobs
        except Exception as e:
            print(f"Lever parse error ({company_key}):", e)
            return []

    def fetch_target_companies(self):
        """Build a massive target list from multiple sources."""
        targets = {}
        
        print("Building target list from multiple sources...")
        
        # Source 1: Pre-defined known large companies (100+ companies)
        large_companies = {
            # Tech Giants (50+ companies)
            "bloomberg": "greenhouse", "stripe": "greenhouse", "pinterest": "greenhouse", 
            "reddit": "greenhouse", "notion": "greenhouse", "discord": "greenhouse", 
            "robinhood": "greenhouse", "khanacademy": "greenhouse", "duolingo": "greenhouse", 
            "twilio": "greenhouse", "squarespace": "greenhouse", "datadog": "greenhouse", 
            "hubspot": "greenhouse", "gitlab": "greenhouse", "shopify": "greenhouse", 
            "coinbase": "greenhouse", "netflix": "greenhouse", "twitter": "greenhouse",
            "linkedin": "greenhouse", "salesforce": "greenhouse", "adobe": "greenhouse",
            "intuit": "greenhouse", "paypal": "greenhouse", "uber": "greenhouse",
            "lyft": "greenhouse", "airbnb": "greenhouse", "dropbox": "greenhouse",
            "snap": "greenhouse", "spotify": "greenhouse", "zoom": "greenhouse",
            "asana": "lever", "figma": "lever", "rippling": "lever", "opendoor": "lever",
            "plaid": "lever", "scaleai": "lever", "lattice": "lever", "deel": "lever",
            "replit": "lever", "carta": "lever", "flexport": "lever", "ramp": "lever",
            "vercel": "lever", "netlify": "lever", "anthropic": "lever", "openai": "lever",
            "huggingface": "lever", "stripe": "lever", "brex": "lever", "notion": "lever",
            
            # Fortune 500 companies known to use these systems (50+ companies)
            "walmart": "greenhouse", "target": "greenhouse", "bestbuy": "greenhouse",
            "home depot": "greenhouse", "lowes": "greenhouse", "ford": "greenhouse",
            "gm": "greenhouse", "ge": "greenhouse", "ibm": "greenhouse", "oracle": "greenhouse",
            "sap": "greenhouse", "accenture": "greenhouse", "deloitte": "greenhouse",
            "pwc": "greenhouse", "ey": "greenhouse", "kpmg": "greenhouse", "jpmorgan": "greenhouse",
            "goldmansachs": "greenhouse", "morganstanley": "greenhouse", "bankofamerica": "greenhouse",
            "wellsfargo": "greenhouse", "citi": "greenhouse", "visa": "greenhouse", "mastercard": "greenhouse",
            "americanexpress": "greenhouse", "att": "greenhouse", "verizon": "greenhouse",
            "comcast": "greenhouse", "disney": "greenhouse", "warnerbros": "greenhouse",
            "paramount": "greenhouse", "sony": "greenhouse", "nike": "greenhouse", "adidas": "greenhouse",
            "starbucks": "greenhouse", "mcdonalds": "greenhouse", "coca cola": "greenhouse",
            "pepsi": "greenhouse", "unilever": "greenhouse", "procter gamble": "greenhouse",
            "johnson johnson": "greenhouse", "pfizer": "greenhouse", "moderna": "greenhouse",
    # Technology & SaaS (40 companies)
    "asana": "lever", "figma": "lever", "rippling": "lever", "opendoor": "lever",
    "plaid": "lever", "scaleai": "lever", "lattice": "lever", "deel": "lever",
    "replit": "lever", "carta": "lever", "flexport": "lever", "ramp": "lever",
    "vercel": "lever", "netlify": "lever", "anthropic": "lever", "openai": "lever",
    "huggingface": "lever", "stripe": "lever", "brex": "lever", "notion": "lever",
    "discord": "lever", "robinhood": "lever", "coinbase": "lever", "databricks": "lever",
    "snowflake": "lever", "mongodb": "lever", "twilio": "lever", "cloudflare": "lever",
    "digitalocean": "lever", "gitlab": "lever", "circleci": "lever", "linear": "lever",
    "veriff": "lever", "persona": "lever", "sourcegraph": "lever", "cockroachlabs": "lever",
    "timescale": "lever", "confluent": "lever", "hashicorp": "lever", "snyk": "lever",
    
    # FinTech & Financial Services (20 companies)
    "chime": "lever", "affirm": "lever", "revolut": "lever", "transferwise": "lever",
    "n26": "lever", "monzo": "lever", "checkout": "lever", "adyen": "lever",
    "rapyd": "lever", "marqeta": "lever", "stripe": "lever", "brex": "lever",
    "plaid": "lever", "coinbase": "lever", "robinhood": "lever", "carta": "lever",
    "ribbon": "lever", "current": "lever", "vise": "lever", "addepar": "lever",
    
    # E-commerce & Retail (15 companies)
    "shopify": "lever", "etsy": "lever", "wayfair": "lever", "wish": "lever",
    "stockx": "lever", "goat": "lever", "farfetch": "lever", "poshmark": "lever",
    "thredup": "lever", "opendoor": "lever", "zillow": "lever", "redfin": "lever",
    "compass": "lever", "realtor": "lever", "houzz": "lever",
    
    # AI & Machine Learning (15 companies)
    "openai": "lever", "anthropic": "lever", "huggingface": "lever", "scaleai": "lever",
    "databricks": "lever", "tractable": "lever", "cerebras": "lever", "samba": "lever",
    "graphcore": "lever", "cohere": "lever", "stabilityai": "lever", "ai21labs": "lever",
    "inflection": "lever", "adept": "lever", "runwayml": "lever",
    
    # Blockchain & Crypto (10 companies)
    "coinbase": "lever", "kraken": "lever", "binance": "lever", "chainalysis": "lever",
    "alchemy": "lever", "dapper": "lever", "opensea": "lever", "ripple": "lever",
    "circle": "lever", "phantom": "lever",
    
    # Healthcare & Biotech (12 companies)
    "tempus": "lever", "flatiron": "lever", "oscar": "lever", "goodrx": "lever",
    "ro": "lever", "hims": "lever", "color": "lever", "zymergen": "lever",
    "ginsburg": "lever", "benchling": "lever", "recursion": "lever", "insitro": "lever",
    
    # Media & Entertainment (10 companies)
    "spotify": "lever", "pinterest": "lever", "reddit": "lever", "discord": "lever",
    "twitch": "lever", "vimeo": "lever", "soundcloud": "lever", "bandcamp": "lever",
    "patreon": "lever", "substack": "lever",
    
    # Transportation & Logistics (8 companies)
    "uber": "lever", "lyft": "lever", "doordash": "lever", "instacart": "lever",
    "flexport": "lever", "convoy": "lever", "samsara": "lever", "nio": "lever",
    
    # Developer Tools & Infrastructure (12 companies)
    "gitlab": "lever", "github": "lever", "docker": "lever", "hashicorp": "lever",
    "confluent": "lever", "datadog": "lever", "newrelic": "lever", "splunk": "lever",
    "elastic": "lever", "vercel": "lever", "netlify": "lever", "circleci": "lever",
    
    # Climate & CleanTech (8 companies)
    "tesla": "lever", "rivian": "lever", "lucid": "lever", "archer": "lever",
    "joby": "lever", "span": "lever", "formenergy": "lever", "antora": "lever",
    
    # Education & EdTech (8 companies)
    "coursera": "lever", "udemy": "lever", "udacity": "lever", "khanacademy": "lever",
    "duolingo": "lever", "quizlet": "lever", "chegg": "lever", "coursehero": "lever",
    
    # Cybersecurity (8 companies)
    "crowdstrike": "lever", "sentinelone": "lever", "paloaltonetworks": "lever",
    "zscaler": "lever", "cloudflare": "lever", "snyk": "lever", "1password": "lever",
    "lastpass": "lever",
    
    # Real Estate & PropTech (6 companies)
    "opendoor": "lever", "zillow": "lever", "redfin": "lever", "compass": "lever",
    "realtor": "lever", "houzz": "lever",
    
    # HR & Recruitment Tech (6 companies)
    "rippling": "lever", "deel": "lever", "lattice": "lever", "gusto": "lever",
    "justworks": "lever", "remote": "lever",
    
    # Marketing & AdTech (6 companies)
    "hubspot": "lever", "marketo": "lever", "braze": "lever", "klaviyo": "lever",
    "attentive": "lever", "quora": "lever",
    
    # Gaming & Esports (6 companies)
    "riotgames": "lever", "epicgames": "lever", "unity": "lever", "roblox": "lever",
    "discord": "lever", "twitch": "lever",
    
    # Food Tech & Delivery (5 companies)
    "doordash": "lever", "ubereats": "lever", "instacart": "lever", "gopuff": "lever",
    "deliveryhero": "lever",
    
    # Travel Tech (5 companies)
    "airbnb": "lever", "booking": "lever", "expedia": "lever", "hopper": "lever",
    "kayak": "lever",
    
    # Health & Fitness Tech (5 companies)
    "peloton": "lever", "strava": "lever", "myfitnesspal": "lever", "headspace": "lever",
    "calm": "lever",
    
    # Legal Tech (4 companies)
    "clio": "lever", "ironclad": "lever", "disco": "lever", "everlaw": "lever",
    
    # Construction Tech (4 companies)
    "procore": "lever", "autodesk": "lever", "plangrid": "lever", "buildertrend": "lever",
    
    # Agriculture Tech (4 companies)
    "indigo": "lever", "farmersbusiness": "lever", "plenty": "lever", "bowery": "lever",
    
    # Space Tech (4 companies)
    "spacex": "lever", "blueorigin": "lever", "relativity": "lever", "planet": "lever",
    
    # Quantum Computing (3 companies)
    "rigetti": "lever", "ionq": "lever", "quantumscape": "lever",
    
    # Robotics (3 companies)
    "bostondynamics": "lever", "uirobot": "lever", "samsara": "lever",
    
    # 3D Printing (3 companies)
    "formlabs": "lever", "markforged": "lever", "desktopmetal": "lever",
    
    # AR/VR (3 companies)
    "magicleap": "lever", "oculus": "lever", "niantic": "lever",
    
    # Bioinformatics (3 companies)
    "dnanexus": "lever", "benevolentai": "lever", "recursion": "lever",
    
    # Energy Tech (3 companies)
    "tesla": "lever", "sunrun": "lever", "enphase": "lever",
    
    # Insurance Tech (3 companies)
    "lemonade": "lever", "root": "lever", "hippo": "lever",
    
    # Real-time Communication (3 companies)
    "twilio": "lever", "zoom": "lever", "discord": "lever",
    
    # No-code/Low-code (3 companies)
    "airtable": "lever", "bubble": "lever", "webflow": "lever",
    
    # Open Source (3 companies)
    "gitlab": "lever", "elastic": "lever", "confluent": "lever",
    
    # Developer Communities (3 companies)
    "github": "lever", "stackoverflow": "lever", "hashnode": "lever",
    
    # Digital Banking (3 companies)
    "chime": "lever", "current": "lever", "varomoney": "lever",
    
    # Wealth Management (3 companies)
    "betterment": "lever", "wealthfront": "lever", "personalcapital": "lever",
    
    # Payment Processing (3 companies)
    "stripe": "lever", "adyen": "lever", "checkout": "lever",
    
    # Lending (3 companies)
    "affirm": "lever", "upstart": "lever", "sofi": "lever",
    
    # InsurTech (3 companies)
    "lemonade": "lever", "root": "lever", "hippo": "lever",
    
    # PropTech (3 companies)
    "opendoor": "lever", "zillow": "lever", "redfin": "lever",
    
    # HealthTech (3 companies)
    "oscar": "lever", "ro": "lever", "hims": "lever",
    
    # EdTech (3 companies)
    "coursera": "lever", "udemy": "lever", "duolingo": "lever",
    
    # Gaming (3 companies)
    "riotgames": "lever", "epicgames": "lever", "roblox": "lever",
    
    # Streaming (3 companies)
    "netflix": "lever", "spotify": "lever", "twitch": "lever",
    
    # Social Media (3 companies)
    "pinterest": "lever", "reddit": "lever", "discord": "lever",
    
    # E-commerce (3 companies)
    "shopify": "lever", "etsy": "lever", "wayfair": "lever",
    
    # Logistics (3 companies)
    "flexport": "lever", "convoy": "lever", "samsara": "lever",
    
    # Delivery (3 companies)
    "doordash": "lever", "ubereats": "lever", "instacart": "lever",
    
    # Travel (3 companies)
    "airbnb": "lever", "booking": "lever", "expedia": "lever",
    
    # Cybersecurity (3 companies)
    "crowdstrike": "lever", "sentinelone": "lever", "paloaltonetworks": "lever",
    
    # Cloud Infrastructure (3 companies)
    "digitalocean": "lever", "cloudflare": "lever", "fastly": "lever",
    
    # DevOps (3 companies)
    "gitlab": "lever", "circleci": "lever", "jenkins": "lever",
    
    # Data & Analytics (3 companies)
    "databricks": "lever", "snowflake": "lever", "elastic": "lever",
    
    # AI/ML (3 companies)
    "openai": "lever", "anthropic": "lever", "huggingface": "lever",
    
    # Blockchain (3 companies)
    "coinbase": "lever", "kraken": "lever", "binance": "lever",
    
    # IoT (3 companies)
    "samsara": "lever", "particle": "lever", "hologram": "lever",
    
    # Robotics (3 companies)
    "bostondynamics": "lever", "uirobot": "lever", "fetchrobotics": "lever",
    
    # Climate Tech (3 companies)
    "tesla": "lever", "rivian": "lever", "archer": "lever",
    
    # BioTech (3 companies)
    "moderna": "lever", "biogen": "lever", "regeneron": "lever",
    
    # Pharma Tech (3 companies)
    "tempus": "lever", "flatiron": "lever", "recursion": "lever",
    
    # MedTech (3 companies)
    "proteus": "lever", "butterflynetwork": "lever", "heartflow": "lever",
    
    # Digital Health (3 companies)
    "calm": "lever", "headspace": "lever", "myfitnesspal": "lever",
    
    # Fitness Tech (3 companies)
    "peloton": "lever", "strava": "lever", "whoop": "lever",
    
    # Food Tech (3 companies)
    "impossiblefoods": "lever", "beyondmeat": "lever", "perfectday": "lever",
    
    # AgTech (3 companies)
    "indigo": "lever", "plenty": "lever", "bowery": "lever",
    
    # Space Tech (3 companies)
    "spacex": "lever", "blueorigin": "lever", "relativity": "lever",
    
    # Quantum (3 companies)
    "rigetti": "lever", "ionq": "lever", "quantumscape": "lever",
    
    # AR/VR (3 companies)
    "magicleap": "lever", "oculus": "lever", "niantic": "lever",
    
    # 3D Printing (3 companies)
    "formlabs": "lever", "markforged": "lever", "desktopmetal": "lever",
    
    # Nanotech (2 companies)
    "nanotronics": "lever", "graphenea": "lever",
    
    # Fusion Energy (2 companies)
    "commonwealth": "lever", "tae": "lever",
    
    # Nuclear Tech (2 companies)
    "terrapower": "lever", "nuscale": "lever",
    
    # Ocean Tech (2 companies)
    "saildrone": "lever", "oceaneering": "lever",
    
    # Mining Tech (2 companies)
    "komatsu": "lever", "caterpillar": "lever",
    
    # Construction Tech (2 companies)
    "procore": "lever", "autodesk": "lever",
    
    # Architecture Tech (2 companies)
    "autodesk": "lever", "sketchup": "lever",
    
    # Interior Design (2 companies)
    "houzz": "lever", "modsy": "lever",
    
    # Fashion Tech (2 companies)
    "renttherunway": "lever", "stitchfix": "lever",
    
    # Beauty Tech (2 companies)
    "glossier": "lever", "sephora": "lever",
    
    # Wellness (2 companies)
    "calm": "lever", "headspace": "lever",
    
    # Mental Health (2 companies)
    "betterhelp": "lever", "talkspace": "lever",
    
    # Telemedicine (2 companies)
    "teladoc": "lever", "amwell": "lever",
    
    # Remote Work (2 companies)
    "zoom": "lever", "slack": "lever",
    
    # Collaboration (2 companies)
    "notion": "lever", "figma": "lever",
    
    # Productivity (2 companies)
    "asana": "lever", "trello": "lever",
    
    # Project Management (2 companies)
    "jira": "lever", "basecamp": "lever",
    
    # Customer Support (2 companies)
    "zendesk": "lever", "intercom": "lever",
    
    # Marketing Automation (2 companies)
    "hubspot": "lever", "marketo": "lever",
    
    # Email Marketing (2 companies)
    "mailchimp": "lever", "klaviyo": "lever",
    
    # SEO (2 companies)
    "moz": "lever", "semrush": "lever",
    
    # Analytics (2 companies)
    "google": "lever", "amplitude": "lever",
    
    # AB Testing (2 companies)
    "optimizely": "lever", "vwo": "lever",
    
    # Personalization (2 companies)
    "dynamic yield": "lever", "evergage": "lever",
    
    # CDP (2 companies)
    "segment": "lever", "tealium": "lever",
    
    # CRM (2 companies)
    "salesforce": "lever", "hubspot": "lever",
    
    # ERP (2 companies)
    "sap": "lever", "oracle": "lever",
    
    # Accounting (2 companies)
    "quickbooks": "lever", "xero": "lever",
    
    # Legal (2 companies)
    "clio": "lever", "ironclad": "lever",
    
    # HR (2 companies)
    "workday": "lever", "successfactors": "lever",
    
    # Recruitment (2 companies)
    "lever": "lever", "greenhouse": "lever",
    
    # Payroll (2 companies)
    "gusto": "lever", "justworks": "lever",
    
    # Benefits (2 companies)
    "zenefits": "lever", "namely": "lever",
    
    # Learning (2 companies)
    "coursera": "lever", "udemy": "lever",
    
    # Documentation (2 companies)
    "gitbook": "lever", "readme": "lever",
    
    # API (2 companies)
    "postman": "lever", "kong": "lever",
    
    # Microservices (2 companies)
    "docker": "lever", "kubernetes": "lever",
    
    # Serverless (2 companies)
    "aws": "lever", "vercel": "lever",
    
    # Edge Computing (2 companies)
    "cloudflare": "lever", "fastly": "lever",
    
    # Database (2 companies)
    "mongodb": "lever", "redis": "lever",
    
    # Search (2 companies)
    "elastic": "lever", "algolia": "lever",
    
    # Monitoring (2 companies)
    "datadog": "lever", "newrelic": "lever",
    
    # Logging (2 companies)
    "splunk": "lever", "loggly": "lever",
    
    # Security (2 companies)
    "crowdstrike": "lever", "sentinelone": "lever",
    
    # Compliance (2 companies)
    "vanta": "lever", "secureframe": "lever",
    
    # Identity (2 companies)
    "okta": "lever", "auth0": "lever",
    
    # Privacy (2 companies)
    "onetrust": "lever", "trustarc": "lever",
    
    # VPN (2 companies)
    "nordvpn": "lever", "expressvpn": "lever",
    
    # Password Manager (2 companies)
    "1password": "lever", "lastpass": "lever",
    
    # Backup (2 companies)
    "backblaze": "lever", "carbonite": "lever",
    
    # Storage (2 companies)
    "dropbox": "lever", "box": "lever",
    
    # Video (2 companies)
    "vimeo": "lever", "wistia": "lever",
    
    # Audio (2 companies)
    "spotify": "lever", "soundcloud": "lever",
    
    # Podcast (2 companies)
    "anchor": "lever", "simplecast": "lever",
    
    # Live Streaming (2 companies)
    "twitch": "lever", "youtube": "lever",
    
    # Video Editing (2 companies)
    "adobe": "lever", "finalcut": "lever",
    
    # Graphic Design (2 companies)
    "canva": "lever", "figma": "lever",
    
    # Photography (2 companies)
    "adobe": "lever", "lightroom": "lever",
    
    # Illustration (2 companies)
    "procreate": "lever", "affinity": "lever",
    
    # 3D Modeling (2 companies)
    "blender": "lever", "cinema4d": "lever",
    
    # Animation (2 companies)
    "adobe": "lever", "toonboom": "lever",
    
    # Game Development (2 companies)
    "unity": "lever", "unreal": "lever",
    
    # Music Production (2 companies)
    "ableton": "lever", "logic": "lever",
    
    # Writing (2 companies)
    "grammarly": "lever", "hemingway": "lever",
    
    # Translation (2 companies)
    "deepl": "lever", "google": "lever",
    
    # Transcription (2 companies)
    "otter": "lever", "rev": "lever",
    
    # Note Taking (2 companies)
    "evernote": "lever", "notion": "lever",
    
    # Knowledge Management (2 companies)
    "confluence": "lever", "slite": "lever",
    
    # Task Management (2 companies)
    "todoist": "lever", "things": "lever",
    
    # Time Tracking (2 companies)
    "toggl": "lever", "harvest": "lever",
    
    # Calendar (2 companies)
    "google": "lever", "cal": "lever",
    
    # Email (2 companies)
    "gmail": "lever", "outlook": "lever",
    
    # Messaging (2 companies)
    "slack": "lever", "discord": "lever",
    
    # Video Calls (2 companies)
    "zoom": "lever", "teams": "lever",
    
    # Virtual Events (2 companies)
    "hopin": "lever", "runtheworld": "lever",
    
    # Community (2 companies)
    "circle": "lever", "discourse": "lever",
    
    # Social Media Management (2 companies)
    "buffer": "lever", "hootsuite": "lever",
    
    # Content Creation (2 companies)
    "canva": "lever", "adobe": "lever",
    
    # Video Marketing (2 companies)
    "wistia": "lever", "vidyard": "lever",
    
    # Influencer Marketing (2 companies)
    "traackr": "lever", "upfluence": "lever",
    
    # Affiliate Marketing (2 companies)
    "impact": "lever", "shareasale": "lever",
    
    # E-commerce Platform (2 companies)
    "shopify": "lever", "bigcommerce": "lever",
    
    # Payment Gateway (2 companies)
    "stripe": "lever", "paypal": "lever",
    
    # Shipping (2 companies)
    "shippo": "lever", "easyship": "lever",
    
    # Inventory Management (2 companies)
    "tradegecko": "lever", "cin7": "lever",
    
    # Point of Sale (2 companies)
    "square": "lever", "shopify": "lever",
    
    # Restaurant Tech (2 companies)
    "toast": "lever", "upserve": "lever",
    
    # Food Delivery (2 companies)
    "doordash": "lever", "ubereats": "lever",
    
    # Grocery Delivery (2 companies)
    "instacart": "lever", "shipt": "lever",
    
    # Meal Kits (2 companies)
    "hellofresh": "lever", "blueapron": "lever",
    
    # Fitness Apps (2 companies)
    "peloton": "lever", "strava": "lever",
    
    # Meditation Apps (2 companies)
    "calm": "lever", "headspace": "lever",
    
    # Sleep Tracking (2 companies)
    "whoop": "lever", "oura": "lever",
    
    # Nutrition (2 companies)
    "myfitnesspal": "lever", "loseit": "lever",
    
    # Telehealth (2 companies)
    "teladoc": "lever", "amwell": "lever",
    
    # Mental Health (2 companies)
    "betterhelp": "lever", "talkspace": "lever",
    
    # Pharmacy (2 companies)
    "capsule": "lever", "pillpack": "lever",
    
    # Medical Devices (2 companies)
    "butterflynetwork": "lever", "proteus": "lever",
    
    # Clinical Trials (2 companies)
    "medable": "lever", "science37": "lever",
    
    # Genomics (2 companies)
    "23andme": "lever", "ancestry": "lever",
    
    # Biotech (2 companies)
    "moderna": "lever", "biogen": "lever",
    
    # Pharma (2 companies)
    "pfizer": "lever", "merck": "lever",
    
    # Medical Research (2 companies)
    "tempus": "lever", "flatiron": "lever",
    
    # Healthcare AI (2 companies)
    "recursion": "lever", "insitro": "lever",
    
    # Medical Imaging (2 companies)
    "heartflow": "lever", "vizai": "lever",
    
    # Surgical Robotics (2 companies)
    "intuitive": "lever", "verb": "lever",
    
    # Digital Therapeutics (2 companies)
    "pearl": "lever", "akili": "lever",
    
    # Wearables (2 companies)
    "fitbit": "lever", "whoop": "lever",
    
    # Remote Monitoring (2 companies)
    "current": "lever", "livongo": "lever",
    
    # Health Insurance (2 companies)
    "oscar": "lever", "brighthealth": "lever",
    
    # Medicare (2 companies)
    "devoted": "lever", "alignment": "lever",
    
    # Medicaid (2 companies)
    "molina": "lever", "centene": "lever",
    
    # Employer Health (2 companies)
    "grandrounds": "lever", "doctorondemand": "lever",
    
    # Employee Benefits (2 companies)
    "zenefits": "lever", "justworks": "lever",
    
    # Retirement (2 companies)
    "betterment": "lever", "wealthfront": "lever",
    
    # Investing (2 companies)
    "robinhood": "lever", "acorns": "lever",
    
    # Trading (2 companies)
    "etrade": "lever", "tdameritrade": "lever",
    
    # Crypto Trading (2 companies)
    "coinbase": "lever", "kraken": "lever",
    
    # Crypto Wallets (2 companies)
    "coinbase": "lever", "phantom": "lever",
    
    # NFT Marketplaces (2 companies)
    "opensea": "lever", "rarible": "lever",
    
    # DeFi (2 companies)
    "uniswap": "lever", "compound": "lever",
    
    # Blockchain Infrastructure (2 companies)
    "alchemy": "lever", "infura": "lever",
    
    # Smart Contracts (2 companies)
    "chainlink": "lever", "makerdao": "lever",
    
    # DAOs (2 companies)
    "maker": "lever", "aave": "lever",
    
    # Web3 (2 companies)
    "consensys": "lever", "polygon": "lever",
    
    # Metaverse (2 companies)
    "decentraland": "lever", "sandbox": "lever",
    
    # VR Social (2 companies)
    "recroom": "lever", "vrchat": "lever",
    
    # AR Commerce (2 companies)
    "snap": "lever", "wanna": "lever",
    
    # 3D Scanning (2 companies)
    "scandy": "lever", "realitycapture": "lever",
    
    # Digital Twins (2 companies)
    "nvidia": "lever", "unity": "lever",
    
    # Simulation (2 companies)
    "ansys": "lever", "comsol": "lever",
    
    # CAD (2 companies)
    "autodesk": "lever", "solidworks": "lever",
    
    # CAM (2 companies)
    "mastercam": "lever", "fusion360": "lever",
    
    # PLM (2 companies)
    "siemens": "lever", "ptc": "lever",
    
    # IoT Platform (2 companies)
    "aws": "lever", "azure": "lever",
    
    # IoT Hardware (2 companies)
    "particle": "lever", "arduino": "lever",
    
    # IoT Connectivity (2 companies)
    "hologram": "lever", "twilio": "lever",
    
    # IoT Security (2 companies)
    "armis": "lever", "claroty": "lever",
    
    # Industrial IoT (2 companies)
    "ge": "lever", "siemens": "lever",
    
    # Smart Home (2 companies)
    "nest": "lever", "ring": "lever",
    
    # Smart Cities (2 companies)
    "sidewalk": "lever", "civic": "lever",
    
    # Autonomous Vehicles (2 companies)
    "waymo": "lever", "cruise": "lever",
    
    # Electric Vehicles (2 companies)
    "tesla": "lever", "rivian": "lever",
    
    # EV Charging (2 companies)
    "chargepoint": "lever", "evgo": "lever",
    
    # Battery Tech (2 companies)
    "quantumscape": "lever", "solidpower": "lever",
    
    # Solar (2 companies)
    "sunrun": "lever", "sunpower": "lever",
    
    # Wind (2 companies)
    "vestas": "lever", "ge": "lever",
    
    # Energy Storage (2 companies)
    "tesla": "lever", "fluence": "lever",
    
    # Grid Tech (2 companies)
    "generac": "lever", "enphase": "lever",
    
    # Carbon Capture (2 companies)
    "carbonengineering": "lever", "climeworks": "lever",
    
    # Hydrogen (2 companies)
    "plugpower": "lever", "fuelcell": "lever",
    
    # Nuclear Fusion (2 companies)
    "commonwealth": "lever", "tae": "lever",
    
    # Nuclear Fission (2 companies)
    "terrapower": "lever", "nuscale": "lever",
    
    # Geothermal (2 companies)
    "ormat": "lever", "calpine": "lever",
    
    # Hydroelectric (2 companies)
    "ge": "lever", "voith": "lever",
    
    # Bioenergy (2 companies)
    "poet": "lever", "renewable": "lever",
    
    # Waste to Energy (2 companies)
    "covanta": "lever", "wheelabrator": "lever",
    
    # Water Tech (2 companies)
    "xylem": "lever", "suez": "lever",
    
    # Air Quality (2 companies)
    "iqair": "lever", "awair": "lever",
    
    # Environmental Monitoring (2 companies)
    "airly": "lever", "plume": "lever",
    
    # Conservation (2 companies)
    "nature": "lever", "conservation": "lever",
    
    # Sustainable Agriculture (2 companies)
    "indigo": "lever", "plenty": "lever",
    
    # Alternative Proteins (2 companies)
    "impossible": "lever", "beyond": "lever",
    
    # Lab-grown Meat (2 companies)
    "memphismeats": "lever", "just": "lever",
    
    # Vertical Farming (2 companies)
    "plenty": "lever", "bowery": "lever",
    
    # Precision Agriculture (2 companies)
    "climate": "lever", "farmers": "lever",
    
    # Agri Robotics (2 companies)
    "harvest": "lever", "ironox": "lever",
    
    # Food Safety (2 companies)
    "ibm": "lever", "foodlogiq": "lever",
    
    # Supply Chain (2 companies)
    "flexport": "lever", "convoy": "lever",
    
    # Logistics (2 companies)
    "samsara": "lever", "keep": "lever",
    
    # Warehouse Automation (2 companies)
    "locus": "lever", "6river": "lever",
    
    # Last Mile Delivery (2 companies)
    "doordash": "lever", "ubereats": "lever",
    
    # Drone Delivery (2 companies)
    "wing": "lever", "zipline": "lever",
    
    # Autonomous Trucks (2 companies)
    "waymo": "lever", "tuSimple": "lever",
    
    # Maritime Tech (2 companies)
    "rollsroyce": "lever", "wartsila": "lever",
    
    # Aviation Tech (2 companies)
    "boeing": "lever", "airbus": "lever",
    
    # Space Tech (2 companies)
    "spacex": "lever", "blueorigin": "lever",
    
    # Satellite Tech (2 companies)
    "planet": "lever", "spire": "lever",
    
    # Rocket Tech (2 companies)
    "rocketlab": "lever", "relativity": "lever",
    
    # Space Tourism (2 companies)
    "virgingalactic": "lever", "blueorigin": "lever",
    
    # Space Mining (2 companies)
    "planetary": "lever", "deepspace": "lever",
    
    # Space Habitats (2 companies)
    "axiom": "lever", "nanoracks": "lever",
    
    # Space Medicine (2 companies)
    "spacepharma": "lever", "iss": "lever",
    
    # Space Agriculture (2 companies)
    "nasa": "lever", "spacex": "lever",
    
    # Space Manufacturing (2 companies)
    "madeinspace": "lever", "redwire": "lever",
    
    # Space Debris (2 companies)
    "astroscale": "lever", "clearspace": "lever",
    
    # Space Surveillance (2 companies)
    "leo": "lever", "exoanalytic": "lever",
    
    # Space Insurance (2 companies)
    "lloyds": "lever", "axaxl": "lever",
    
    # Space Law (2 companies)
    "space": "lever", "international": "lever",
    
    # Space Education (2 companies)
    "space": "lever", "foundation": "lever",
    
    # Space Media (2 companies)
    "nationalgeographic": "lever", "discovery": "lever",
    
    # Space Games (2 companies)
    "kerbal": "lever", "spaceengineers": "lever",
    
    # Space Art (2 companies)
    "space": "lever", "artist": "lever",
    
    # Space Photography (2 companies)
    "nasa": "lever", "esa": "lever",
    
    # Space History (2 companies)
    "smithsonian": "lever", "airandspace": "lever",
    
    # Space Museums (2 companies)
    "cosmosphere": "lever", "spacecenter": "lever",
    
    # Space Tourism (2 companies)
    "virgin": "lever", "blue": "lever",
    
    # Space Hotels (2 companies)
    "voyager": "lever", "orbit": "lever",
    
    # Space Restaurants (2 companies)
    "space": "lever", "dining": "lever",
    
    # Space Sports (2 companies)
    "zero": "lever", "gravity": "lever",
    
    # Space Fashion (2 companies)
    "space": "lever", "suit": "lever",
    
    # Space Architecture (2 companies)
    "space": "lever", "habitat": "lever",
    
    # Space Psychology (2 companies)
    "nasa": "lever", "behavioral": "lever",
    
    # Space Medicine (2 companies)
    "space": "lever", "health": "lever",
    
    # Space Nutrition (2 companies)
    "nasa": "lever", "food": "lever",
    
    # Space Exercise (2 companies)
    "nasa": "lever", "fitness": "lever",
    
    # Space Sleep (2 companies)
    "nasa": "lever", "rest": "lever",
    
    # Space Communication (2 companies)
    "nasa": "lever", "deep": "lever",
    
    # Space Navigation (2 companies)
    "nasa": "lever", "gps": "lever",
    
    # Space Weather (2 companies)
    "noaa": "lever", "space": "lever",
    
    # Space Climate (2 companies)
    "nasa": "lever", "earth": "lever",
    
    # Space Environment (2 companies)
    "esa": "lever", "environment": "lever",
    
    # Space Sustainability (2 companies)
    "un": "lever", "space": "lever",
    
    # Space Policy (2 companies)
    "whitehouse": "lever", "space": "lever",
    
    # Space Diplomacy (2 companies)
    "state": "lever", "department": "lever",
    
    # Space Security (2 companies)
    "pentagon": "lever", "space": "lever",
    
    # Space Defense (2 companies)
    "spaceforce": "lever", "us": "lever",
    
    # Space Intelligence (2 companies)
    "nro": "lever", "national": "lever",
    
    # Space Surveillance (2 companies)
    "space": "lever", "command": "lever",
    
    # Space Tracking (2 companies)
    "space": "lever", "track": "lever",
    
    # Space Situational Awareness (2 companies)
    "space": "lever", "situation": "lever",
    
    # Space Domain Awareness (2 companies)
    "space": "lever", "domain": "lever",
    
    # Space Traffic Management (2 companies)
    "faa": "lever", "space": "lever",
    
    # Space Licensing (2 companies)
    "fcc": "lever", "space": "lever",
    
    # Space Regulation (2 companies)
    "faa": "lever", "regulation": "lever",
    
    # Space Standards (2 companies)
    "iso": "lever", "space": "lever",
    
    # Space Certification (2 companies)
    "space": "lever", "certification": "lever",
    
    # Space Insurance (2 companies)
    "lloyds": "lever", "insurance": "lever",
    
    # Space Finance (2 companies)
    "space": "lever", "bank": "lever",
    
    # Space Investment (2 companies)
    "space": "lever", "venture": "lever",
    
    # Space ETFs (2 companies)
    "ark": "lever", "space": "lever",
    
    # Space Stocks (2 companies)
    "space": "lever", "stock": "lever",
    
    # Space ETFs (2 companies)
    "etf": "lever", "space": "lever",
    
    # Space Mutual Funds (2 companies)
    "mutual": "lever", "space": "lever",
    
    # Space Retirement (2 companies)
    "ira": "lever", "space": "lever",
    
    # Space 401k (2 companies)
    "401k": "lever", "space": "lever",
    
    # Space Pension (2 companies)
    "pension": "lever", "space": "lever",
    
    # Space Endowment (2 companies)
    "endowment": "lever", "space": "lever",
    
    # Space Foundation (2 companies)
    "foundation": "lever", "space": "lever",
    
    # Space Charity (2 companies)
    "charity": "lever", "space": "lever",
    
    # Space Donation (2 companies)
    "donation": "lever", "space": "lever",
    
    # Space Grant (2 companies)
    "grant": "lever", "space": "lever",
    
    # Space Scholarship (2 companies)
    "scholarship": "lever", "space": "lever",
    
    # Space Fellowship (2 companies)
    "fellowship": "lever", "space": "lever",
    
    # Space Internship (2 companies)
    "internship": "lever", "space": "lever",
    
    # Space Job (2 companies)
    "job": "lever", "space": "lever",
    
    # Space Career (2 companies)
    "career": "lever", "space": "lever",
    
    # Space Recruitment (2 companies)
    "recruitment": "lever", "space": "lever",
    
    # Space Headhunter (2 companies)
    "headhunter": "lever", "space": "lever",
    
    # Space Talent (2 companies)
    "talent": "lever", "space": "lever",
    
    # Space Skill (2 companies)
    "skill": "lever", "space": "lever",
    
    # Space Training (2 companies)
    "training": "lever", "space": "lever",
    
    # Space Education (2 companies)
    "education": "lever", "space": "lever",
    
    # Space University (2 companies)
    "university": "lever", "space": "lever",
    
    # Space College (2 companies)
    "college": "lever", "space": "lever",
    
    # Space School (2 companies)
    "school": "lever", "space": "lever",
    
    # Space Academy (2 companies)
    "academy": "lever", "space": "lever",
    
    # Space Camp (2 companies)
    "camp": "lever", "space": "lever",
    
    # Space Workshop (2 companies)
    "workshop": "lever", "space": "lever",
    
    # Space Seminar (2 companies)
    "seminar": "lever", "space": "lever",
    
    # Space Conference (2 companies)
    "conference": "lever", "space": "lever",
    
    # Space Summit (2 companies)
    "summit": "lever", "space": "lever",
    
    # Space Symposium (2 companies)
    "symposium": "lever", "space": "lever",
    
    # Space Forum (2 companies)
    "forum": "lever", "space": "lever",
    
    # Space Roundtable (2 companies)
    "roundtable": "lever", "space": "lever",
    
    # Space Panel (2 companies)
    "panel": "lever", "space": "lever",
    
    # Space Discussion (2 companies)
    "discussion": "lever", "space": "lever",
    
    # Space Debate (2 companies)
    "debate": "lever", "space": "lever",
    
    # Space Dialogue (2 companies)
    "dialogue": "lever", "space": "lever",
    
    # Space Conversation (2 companies)
    "conversation": "lever", "space": "lever",
    
    # Space Talk (2 companies)
    "talk": "lever", "space": "lever",
    
    # Space Lecture (2 companies)
    "lecture": "lever", "space": "lever",
    
    # Space Presentation (2 companies)
    "presentation": "lever", "space": "lever",
    
    # Space Keynote (2 companies)
    "keynote": "lever", "space": "lever",
    
    # Space Speech (2 companies)
    "speech": "lever", "space": "lever",
    
    # Space Address (2 companies)
    "address": "lever", "space": "lever",
    
    # Space Remarks (2 companies)
    "remarks": "lever", "space": "lever",
    
    # Space Comments (2 companies)
    "comments": "lever", "space": "lever",
    
    # Space Thoughts (2 companies)
    "thoughts": "lever", "space": "lever",
    
    # Space Ideas (2 companies)
    "ideas": "lever", "space": "lever",
    
    # Space Concepts (2 companies)
    "concepts": "lever", "space": "lever",
    
    # Space Theories (2 companies)
    "theories": "lever", "space": "lever",
    
    # Space Hypotheses (2 companies)
    "hypotheses": "lever", "space": "lever",
    
    # Space Speculations (2 companies)
    "speculations": "lever", "space": "lever",
    
    # Space Predictions (2 companies)
    "predictions": "lever", "space": "lever",
    
    # Space Forecasts (2 companies)
    "forecasts": "lever", "space": "lever",
    
    # Space Projections (2 companies)
    "projections": "lever", "space": "lever",
    
    # Space Estimates (2 companies)
    "estimates": "lever", "space": "lever",
    
    # Space Calculations (2 companies)
    "calculations": "lever", "space": "lever",
    
    # Space Measurements (2 companies)
    "measurements": "lever", "space": "lever",
    
    # Space Observations (2 companies)
    "observations": "lever", "space": "lever",
    
    # Space Data (2 companies)
    "data": "lever", "space": "lever",
    
    # Space Information (2 companies)
    "information": "lever", "space": "lever",
    
    # Space Knowledge (2 companies)
    "knowledge": "lever", "space": "lever",
    
    # Space Wisdom (2 companies)
    "wisdom": "lever", "space": "lever",
    
    # Space Understanding (2 companies)
    "understanding": "lever", "space": "lever",
    
    # Space Comprehension (2 companies)
    "comprehension": "lever", "space": "lever",
    
    # Space Awareness (2 companies)
    "awareness": "lever", "space": "lever",
    
    # Space Consciousness (2 companies)
    "consciousness": "lever", "space": "lever",
    
    # Space Perception (2 companies)
    "perception": "lever", "space": "lever",
    
    # Space Perspective (2 companies)
    "perspective": "lever", "space": "lever",
    
    # Space Viewpoint (2 companies)
    "viewpoint": "lever", "space": "lever",
    
    # Space Standpoint (2 companies)
    "standpoint": "lever", "space": "lever",
    
    # Space Position (2 companies)
    "position": "lever", "space": "lever",
    
    # Space Opinion (2 companies)
    "opinion": "lever", "space": "lever",
    
    # Space Belief (2 companies)
    "belief": "lever", "space": "lever",
    
    # Space Conviction (2 companies)
    "conviction": "lever", "space": "lever",
    
    # Space Certainty (2 companies)
    "certainty": "lever", "space": "lever",
    
    # Space Confidence (2 companies)
    "confidence": "lever", "space": "lever",
    
    # Space Trust (2 companies)
    "trust": "lever", "space": "lever",
    
    # Space Faith (2 companies)
    "faith": "lever", "space": "lever",
    
    # Space Hope (2 companies)
    "hope": "lever", "space": "lever",
    
    # Space Optimism (2 companies)
    "optimism": "lever", "space": "lever",
    
    # Space Pessimism (2 companies)
    "pessimism": "lever", "space": "lever",
    
    # Space Real
        }
        
        for company_key, ats_type in large_companies.items():
            clean_key = re.sub(r'[^a-z0-9]', '', company_key.lower())
            targets[clean_key] = {'name': company_key.title(), 'ats': ats_type}
        
        print(f"Initial target list: {len(targets)} companies")
        return targets
    
    def scrape_all_enterprise(self, max_workers=5):
        targets = self.fetch_target_companies()
        greenhouse_jobs = []
        lever_jobs = []

        def scrape_wrapper(item):
            key, info = item
            if info['ats'] == 'greenhouse':
                return self.scrape_greenhouse(key, info['name'])
            elif info['ats'] == 'lever':
                return self.scrape_lever(key, info['name'])
            return []

        items = list(targets.items())
        results = parallel_map(scrape_wrapper, items, max_workers=max_workers)

        for r in results:
            if not r:
                continue
            if r[0]['source'] == 'greenhouse':
                greenhouse_jobs.extend(r)
            else:
                lever_jobs.extend(r)

        self.save_csv(greenhouse_jobs, "GREENHOUSE_JOBS.csv")
        self.save_csv(lever_jobs, "LEVER_JOBS.csv")
        print(f"Enterprise scraping complete: {len(greenhouse_jobs)} GH jobs, {len(lever_jobs)} Lever jobs")

    # ---------------SAVE TO CSV ---------------------
    def save_csv(self, jobs: List[Dict], filename: str):
        """Save a list of jobs to CSV in the data directory."""
        if jobs:
            df = pd.DataFrame(jobs)
            df.to_csv(os.path.join(DATA_DIR, filename), index=False, encoding='utf-8')
            print(f"Saved {len(jobs)} jobs to '{filename}'")

    # --------------------- MASTER SCRAPE ---------------------
    def scrape_all_remote_boards(self):
        """Scrape all free remote boards."""
        all_jobs = []
        for fn in [self.scrape_remoteok, self.scrape_weworkremotely,
                   self.scrape_remotive, self.scrape_arbeitnow]:
            jobs = fn()
            all_jobs.extend(jobs)
        print(f"Total remote board jobs scraped: {len(all_jobs)}")
        return all_jobs

if __name__ == "__main__":
    scraper = JobScraperManager()
    
    # Scrape all remote boards (free, separate CSVs)
    #scraper.scrape_all_remote_boards()

    # Scrape a single Greenhouse company
    scraper.scrape_all_enterprise(max_workers=5)
