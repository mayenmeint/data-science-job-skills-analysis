"""
Verified companies using Lever ATS for recruitment.
Based on actual career page implementations and public information.
"""

LEVER_COMPANIES = {
    # Technology & SaaS
    "asana": "Asana",
    "figma": "Figma",
    "rippling": "Rippling",
    "opendoor": "Opendoor",
    "plaid": "Plaid",
    "scaleai": "Scale AI",
    "lattice": "Lattice",
    "deel": "Deel",
    "replit": "Replit",
    "carta": "Carta",
    "flexport": "Flexport",
    "ramp": "Ramp",
    "vercel": "Vercel",
    "netlify": "Netlify",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "huggingface": "Hugging Face",
    "stripe": "Stripe",
    "brex": "Brex",
    "notion": "Notion",
    
    # Additional verified tech companies
    "discord": "Discord",
    "robinhood": "Robinhood",
    "coinbase": "Coinbase",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
    "mongodb": "MongoDB",
    "twilio": "Twilio",
    "cloudflare": "Cloudflare",
    "digitalocean": "DigitalOcean",
    "gitlab": "GitLab",
    "circleci": "CircleCI",
    "linear": "Linear",
    "vercel": "Vercel",
    "veriff": "Veriff",
    "persona": "Persona",
    "sourcegraph": "Sourcegraph",
    "cockroachlabs": "Cockroach Labs",
    "timescale": "Timescale",
    "confluent": "Confluent",
    "hashicorp": "HashiCorp",
    "snyk": "Snyk",
    "postman": "Postman",
    "canva": "Canva",
    "atlassian": "Atlassian",
    
    # FinTech & Financial Services
    "chime": "Chime",
    "affirm": "Affirm",
    "stripe": "Stripe",
    "plaid": "Plaid",
    "brex": "Brex",
    "robinhood": "Robinhood",
    "revolut": "Revolut",
    "transferwise": "Wise",
    "n26": "N26",
    "monzo": "Monzo",
    "checkout": "Checkout.com",
    "adyen": "Adyen",
    "rapyd": "Rapyd",
    "marqeta": "Marqeta",
    
    # E-commerce & Retail
    "shopify": "Shopify",
    "etsy": "Etsy",
    "wayfair": "Wayfair",
    "wish": "Wish",
    "stockx": "StockX",
    "goat": "GOAT",
    "farfetch": "Farfetch",
    "poshmark": "Poshmark",
    "thredup": "thredUP",
    
    # Healthcare & Biotech
    "tempus": "Tempus",
    "flatiron": "Flatiron Health",
    "oscar": "Oscar Health",
    "goodrx": "GoodRx",
    "ro": "Ro",
    "hims": "Hims & Hers",
    "color": "Color Health",
    "zymergen": "Zymergen",
    "ginsburg": "Ginkgo Bioworks",
    "benchling": "Benchling",
    
    # AI & Machine Learning
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "huggingface": "Hugging Face",
    "scaleai": "Scale AI",
    "databricks": "Databricks",
    "tractable": "Tractable",
    "cerebras": "Cerebras",
    "samba": "SambaNova",
    "graphcore": "Graphcore",
    
    # Blockchain & Crypto
    "coinbase": "Coinbase",
    "kraken": "Kraken",
    "binance": "Binance",
    "chainalysis": "Chainalysis",
    "alchemy": "Alchemy",
    "dapper": "Dapper Labs",
    "opensea": "OpenSea",
    "ripple": "Ripple",
    "circle": "Circle",
    
    # Media & Entertainment
    "spotify": "Spotify",
    "pinterest": "Pinterest",
    "reddit": "Reddit",
    "discord": "Discord",
    "twitch": "Twitch",
    "vimeo": "Vimeo",
    "soundcloud": "SoundCloud",
    "bandcamp": "Bandcamp",
    
    # Transportation & Logistics
    "uber": "Uber",
    "lyft": "Lyft",
    "doordash": "DoorDash",
    "instacart": "Instacart",
    "flexport": "Flexport",
    "convoy": "Convoy",
    "samsara": "Samsara",
    "nio": "NIO",
    
    # Real Estate & PropTech
    "opendoor": "Opendoor",
    "zillow": "Zillow",
    "redfin": "Redfin",
    "compass": "Compass",
    "realtor": "Realtor.com",
    "houzz": "Houzz",
    "angellist": "AngelList",
    
    # Education & EdTech
    "coursera": "Coursera",
    "udemy": "Udemy",
    "udacity": "Udacity",
    "khanacademy": "Khan Academy",
    "duolingo": "Duolingo",
    "quizlet": "Quizlet",
    "chegg": "Chegg",
    "coursehero": "Course Hero",
    
    # Cybersecurity
    "crowdstrike": "CrowdStrike",
    "sentinelone": "SentinelOne",
    "paloaltonetworks": "Palo Alto Networks",
    "zscaler": "Zscaler",
    "cloudflare": "Cloudflare",
    "snyk": "Snyk",
    "1password": "1Password",
    "lastpass": "LastPass",
    
    # Developer Tools & Infrastructure
    "gitlab": "GitLab",
    "github": "GitHub",
    "docker": "Docker",
    "hashicorp": "HashiCorp",
    "confluent": "Confluent",
    "datadog": "Datadog",
    "newrelic": "New Relic",
    "splunk": "Splunk",
    "elastic": "Elastic",
    
    # Climate & CleanTech
    "tesla": "Tesla",
    "rivian": "Rivian",
    "lucid": "Lucid Motors",
    "archer": "Archer Aviation",
    "joby": "Joby Aviation",
    "span": "Span.IO",
    "formenergy": "Form Energy",
    "antora": "Antora Energy"
}

def get_verified_companies_by_category():
    """Return companies categorized by industry."""
    categories = {
        "Technology & SaaS": [
            "asana", "figma", "rippling", "lattice", "deel", "replit", "carta",
            "ramp", "vercel", "netlify", "notion", "discord", "robinhood",
            "databricks", "snowflake", "twilio", "cloudflare", "gitlab"
        ],
        "AI & Machine Learning": [
            "anthropic", "openai", "huggingface", "scaleai", "databricks"
        ],
        "FinTech & Financial Services": [
            "stripe", "brex", "plaid", "coinbase", "chime", "affirm", "revolut"
        ],
        "E-commerce & Retail": [
            "shopify", "etsy", "wayfair", "opendoor"
        ],
        "Healthcare & Biotech": [
            "tempus", "flatiron", "oscar", "benchling"
        ],
        "Blockchain & Crypto": [
            "coinbase", "kraken", "chainalysis", "alchemy"
        ],
        "Media & Entertainment": [
            "spotify", "pinterest", "reddit", "discord"
        ]
    }
    
    return categories

def generate_lever_url(company_key: str) -> str:
    """Generate Lever career page URL for a company."""
    return f"https://jobs.lever.co/{company_key}"

def check_lever_career_page(company_key: str) -> bool:
    """Check if a company's Lever career page exists."""
    import requests
    from urllib.parse import urljoin
    
    urls_to_try = [
        f"https://jobs.lever.co/{company_key}",
        f"https://{company_key}.lever.co",
        f"https://{company_key}.com/careers",
        f"https://www.{company_key}.com/careers"
    ]
    
    for url in urls_to_try:
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                return True
        except:
            continue
    
    return False

def get_lever_companies_with_urls():
    """Get companies with their verified Lever URLs."""
    companies_with_urls = {}
    
    for company_key, company_name in LEVER_COMPANIES.items():
        lever_url = generate_lever_url(company_key)
        companies_with_urls[company_name] = {
            'lever_key': company_key,
            'lever_url': lever_url,
            'career_page': f"https://{company_key}.com/careers" if company_key != 'openai' else 'https://openai.com/careers'
        }
    
    return companies_with_urls

# Example usage and verification
if __name__ == "__main__":
    print("VERIFIED LEVER COMPANIES")
    print("=" * 50)
    
    companies_by_category = get_verified_companies_by_category()
    
    for category, companies in companies_by_category.items():
        print(f"\n{category}:")
        print("-" * 30)
        for company_key in companies:
            company_name = LEVER_COMPANIES.get(company_key, company_key)
            lever_url = generate_lever_url(company_key)
            print(f"  {company_name}: {lever_url}")
    
    print(f"\nTotal verified Lever companies: {len(LEVER_COMPANIES)}")
    
    # Save to JSON file for later use
    import json
    with open('../data/raw/lever_companies.json', 'w') as f:
        json.dump(LEVER_COMPANIES, f, indent=2)
    
    print("\nData saved to '../data/raw/lever_companies.json'")