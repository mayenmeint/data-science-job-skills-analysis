import requests
import pandas as pd
import time
import os

BASE_URL = "https://api.found.dev/api/open/jobs"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_jobs(page=41, skill="Data Science", ai=True):
    """Fetch jobs for a given page and skill across all companies."""
    params = {
        "page": page,
        "skill": skill,
        "ai": str(ai).lower()
    }
    resp = requests.get(BASE_URL, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()

def scrape_in_batches(skill="Data Science", pages_per_batch=20, ai=True, delay=1):
    """Scrape jobs in batches of N pages and save each batch as CSV until no jobs left."""
    os.makedirs("data/raw", exist_ok=True)
    
    batch_num = 3
    page = 41
    total_jobs = 0

    while True:
        all_jobs = []
        print(f"\n🚀 Starting batch {batch_num} (pages {page} → {page + pages_per_batch - 1})")

        for i in range(pages_per_batch):
            print(f"Fetching page {page}...")
            data = fetch_jobs(page=page, skill=skill, ai=ai)
            jobs = data.get("jobs", [])
            
            if not jobs:  # stop if no jobs found
                print("❌ No more jobs found. Stopping.")
                if all_jobs:  # save remaining jobs in this batch
                    df = pd.DataFrame(all_jobs)
                    out_path = f"data/raw/jobs_batch_{batch_num}.csv"
                    df.to_csv(out_path, index=False)
                    print(f"✅ Saved {len(df)} jobs to {out_path}")
                return total_jobs
            
            for entry in jobs:
                job = entry.get("job", {})
                company_info = entry.get("company", {})

                record = {
                    "title": job.get("title"),
                    "company": company_info.get("name"),
                    "city": job.get("city"),
                    "country": job.get("country"),
                    "location": f"{job.get('city')}, {job.get('country')}",
                    "skills": job.get("skills"),
                    "type": job.get("type"),
                    "salary": job.get("salary"),
                    "salary_min": job.get("salary_min"),
                    "salary_max": job.get("salary_max"),
                    "published": job.get("published"),
                    "ai": job.get("ai"),
                }
                all_jobs.append(record)

            total_jobs += len(jobs)
            page += 1
            time.sleep(delay)

        # Save after every batch
        df = pd.DataFrame(all_jobs)
        out_path = f"data/raw/jobs_batch_{batch_num}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Saved {len(df)} jobs to {out_path}")
        
        batch_num += 1


if __name__ == "__main__":
    total = scrape_in_batches(skill="Data Science", pages_per_batch=20, ai=True, delay=1)
    print(f"\n🎉 Finished scraping. Total jobs collected: {total}")
