import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataScienceJobsAnalyzer:
    """
    Analyze data science job market trends from cleaned job data
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the analyzer with data directory paths
        
        Args:
            data_dir: Root data directory path
        """
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.data_dir.parent / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        # Create directories if they don't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        plt.style.use('default')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Load cleaned data from interim directory
        
        Returns:
            Cleaned DataFrame
        """
        interim_files = list(self.interim_dir.glob("*.csv"))
        if not interim_files:
            raise FileNotFoundError("No cleaned data found in interim directory")
        
        df = pd.read_csv(interim_files[0])
        logger.info(f"✅ Loaded {len(df)} cleaned records")
        return df
    
    def analyze_skill_frequency(self, df: pd.DataFrame, top_n: int = 20) -> Dict:
        """
        Analyze frequency of individual skills - FASTEST VERSION using pandas explode
        """
        logger.info("🔧 Analyzing skill frequency (fastest version)...")
        
        # Create a copy to avoid modifying original
        temp_df = df[['skills_parsed']].copy()
        total_jobs = len(temp_df)
        
        # Convert string representations to lists
        def convert_skills(skills):
            if pd.isna(skills):
                return []
            if isinstance(skills, list):
                return skills
            try:
                if skills.startswith('['):
                    return eval(skills)
                else:
                    return [s.strip() for s in skills.split(',')]
            except:
                return []
        
        temp_df['skills_list'] = temp_df['skills_parsed'].apply(convert_skills)
        
        # Use explode to create one row per skill - SUPER FAST
        exploded_df = temp_df.explode('skills_list')
        
        # Drop NaN values from exploded skills
        skills_series = exploded_df['skills_list'].dropna()
        
        # Count frequencies using value_counts (optimized C code)
        skill_value_counts = skills_series.value_counts()
        total_skills = len(skills_series)
        
        # Calculate prevalence
        skill_prevalence = {}
        for skill, count in skill_value_counts.head(top_n * 2).items():  # Check more than needed
            # For prevalence, count distinct jobs containing the skill
            job_count = len(exploded_df[exploded_df['skills_list'] == skill]['skills_parsed'].unique())
            skill_prevalence[skill] = (job_count / total_jobs) * 100
        
        # Get top skills
        top_skills = list(skill_value_counts.head(top_n).items())
        
        return {
            'total_skills_mentioned': total_skills,
            'unique_skills': len(skill_value_counts),
            'avg_skills_per_job': total_skills / total_jobs,
            'top_skills': top_skills,
            'skill_prevalence': skill_prevalence,
            'skill_counts': skill_value_counts.head(top_n * 3).to_dict()  # Store more for other analyses
        }
    
    def plot_top_skills(self, skill_analysis: Dict, figsize: Tuple = (18, 6), 
                   show_scatter: bool = True, top_n: int = 15):
        """
        Create visualization for top skills with optional scatter plot
        
        Args:
            skill_analysis: Dictionary from analyze_skill_frequency
            figsize: Figure size
            show_scatter: Whether to show the scatter plot
            top_n: Number of top skills to display
        """
        if show_scatter:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get top skills by mentions and by prevalence
        top_by_mentions = skill_analysis['top_skills'][:top_n]
        top_by_prevalence = sorted(
            [(skill, skill_analysis['skill_prevalence'][skill]) 
            for skill in skill_analysis['skill_prevalence']],
            key=lambda x: x[1], 
            reverse=True
        )[:top_n]
        
        # Plot 1: Frequency of top skills by mentions
        skills_mentions, counts = zip(*top_by_mentions)
        bars1 = ax1.barh(skills_mentions, counts, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(skills_mentions))))
        ax1.set_xlabel('Number of Mentions')
        ax1.set_title('Top Skills by Frequency\n(Total Mentions)')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()  # Highest at top
        
        # Plot 2: Prevalence of top skills
        skills_prevalence, prevalence_vals = zip(*top_by_prevalence)
        bars2 = ax2.barh(skills_prevalence, prevalence_vals, color=plt.cm.Greens(np.linspace(0.4, 0.8, len(skills_prevalence))))
        ax2.set_xlabel('Percentage of Jobs (%)')
        ax2.set_title('Top Skills by Prevalence\n(% of Jobs Requiring Skill)')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # Add value labels to both bar charts
        for ax, bars, values in [(ax1, bars1, counts), (ax2, bars2, prevalence_vals)]:
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.0f}{"%" if ax == ax2 else ""}', 
                    va='center', fontsize=9, fontweight='bold')
        
        # Plot 3: Scatter plot - Mentions vs Prevalence
        if show_scatter:
            all_skills = list(skill_analysis['skill_counts'].keys())
            mentions = [skill_analysis['skill_counts'].get(skill, 0) for skill in all_skills]
            prevalence = [skill_analysis['skill_prevalence'].get(skill, 0) for skill in all_skills]
            
            # Create bubble sizes based on combined importance
            bubble_sizes = np.sqrt(np.array(mentions) * np.array(prevalence)) * 3
            
            scatter = ax3.scatter(prevalence, mentions, s=bubble_sizes, alpha=0.7,
                                c=np.log1p(mentions), cmap='plasma', edgecolors='black', linewidth=0.5)
            ax3.set_xlabel('Prevalence (% of Jobs)')
            ax3.set_ylabel('Number of Mentions')
            ax3.set_title('Skill Importance Matrix\n(Bubble Size = Overall Impact)')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('log(Mentions + 1)')
            
            # Add correlation line
            z = np.polyfit(prevalence, mentions, 1)
            p = np.poly1d(z)
            ax3.plot(prevalence, p(prevalence), "r--", alpha=0.8, linewidth=1)
            
            # Calculate and display R-squared
            correlation = np.corrcoef(prevalence, mentions)[0,1]
            ax3.text(0.05, 0.95, f'R² = {correlation**2:.3f}', 
                    transform=ax3.transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'skills_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Enhanced skills visualization saved")
    
    def analyze_skill_categories(self, df: pd.DataFrame) -> Dict:
        """
        Analyze distribution of skills across categories
        
        Args:
            df: DataFrame with skill category columns
            
        Returns:
            Dictionary with category analysis
        """
        logger.info("📚 Analyzing skill categories...")
        
        # Get skill category columns
        category_cols = [col for col in df.columns if col.startswith('has_')]
        
        category_analysis = {}
        for category in category_cols:
            category_name = category.replace('has_', '').replace('_', ' ')
            count = df[category].sum()
            percentage = (count / len(df)) * 100
            category_analysis[category_name] = {
                'count': count,
                'percentage': percentage,
                'jobs': count
            }
        
        return category_analysis
    
    def plot_skill_categories(self, category_analysis: Dict, figsize: Tuple = (10, 6)):
        """
        Visualize skill category distribution
        
        Args:
            category_analysis: Dictionary from analyze_skill_categories
            figsize: Figure size
        """
        categories = list(category_analysis.keys())
        percentages = [cat_data['percentage'] for cat_data in category_analysis.values()]
        
        plt.figure(figsize=figsize)
        bars = plt.barh(categories, percentages, color=self.colors[:len(categories)])
        plt.xlabel('Percentage of Jobs (%)')
        plt.title('Prevalence of Skill Categories in Data Science Jobs')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for bar, percentage in zip(bars, percentages):
            plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{percentage:.1f}%', va='center')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'skill_categories.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Skill categories visualization saved")
    
    def analyze_skill_combinations(self, df: pd.DataFrame, top_n: int = 10) -> Dict:
        """
        Analyze most common skill combinations
        
        Args:
            df: DataFrame with skills_parsed column
            top_n: Number of top combinations to analyze
            
        Returns:
            Dictionary with skill combination analysis
        """
        logger.info("🔄 Analyzing skill combinations...")
        
        # Get skill combinations for each job
        combinations = []
        for skills in df['skills_parsed'].dropna():
            if isinstance(skills, str):
                try:
                    skills = eval(skills)
                except:
                    continue
            
            if len(skills) >= 2:
                # Sort skills to ensure consistent ordering
                sorted_skills = sorted(skills)
                # Create pairs of skills
                for i in range(len(sorted_skills)):
                    for j in range(i + 1, len(sorted_skills)):
                        combinations.append((sorted_skills[i], sorted_skills[j]))
        
        # Count combination frequencies
        combo_counts = Counter(combinations)
        top_combinations = combo_counts.most_common(top_n)
        
        return {
            'total_combinations': len(combinations),
            'unique_combinations': len(combo_counts),
            'top_combinations': top_combinations
        }
    
    def analyze_skills_by_seniority(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how skills vary by seniority level
        
        Args:
            df: DataFrame with seniority_level and skills_parsed columns
            
        Returns:
            Dictionary with seniority-based skill analysis
        """
        logger.info("👨‍💼 Analyzing skills by seniority...")
        
        if 'seniority_level' not in df.columns:
            logger.warning("❌ Seniority level data not available")
            return {}
        
        seniority_analysis = {}
        seniority_levels = df['seniority_level'].dropna().unique()
        
        for level in seniority_levels:
            level_data = df[df['seniority_level'] == level]
            
            # Flatten skills for this seniority level
            all_skills = []
            for skills_list in level_data['skills_parsed'].dropna():
                if isinstance(skills_list, str):
                    try:
                        skills_list = eval(skills_list)
                    except:
                        continue
                all_skills.extend(skills_list)
            
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            seniority_analysis[level] = {
                'job_count': len(level_data),
                'top_skills': top_skills,
                'avg_skills_per_job': len(all_skills) / len(level_data) if len(level_data)>0 else 0
            }
        
        return seniority_analysis
    
    def analyze_skills_by_location(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how skills vary by location
        
        Args:
            df: DataFrame with location_category and skills_parsed columns
            
        Returns:
            Dictionary with location-based skill analysis
        """
        logger.info("🌍 Analyzing skills by location...")
        
        location_analysis = {}
        locations = df['country'].dropna().unique()
        
        for location in locations:
            location_data = df[df['country'] == location]
            
            # Flatten skills for this location
            all_skills = []
            for skills_list in location_data['skills_parsed'].dropna():
                if isinstance(skills_list, str):
                    try:
                        skills_list = eval(skills_list)
                    except:
                        continue
                all_skills.extend(skills_list)
            
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(10)
            
            location_analysis[location] = {
                'job_count': len(location_data),
                'top_skills': top_skills,
                'avg_skills_per_job': len(all_skills) / len(location_data) if len(location_data)>0 else 0
            }
        
        return location_analysis
    
    def analyze_skills_correlation_between_titles(self, df: pd.DataFrame, title_1:str='', title_2:str='') -> tuple:
        """
        Analyze correlation between job titles and specific skills
        
        Args:
            df: DataFrame with titles and skills_categorized columns
            title_1: string with the name of the first title
            title_2: string with the name of the second title
            
        Returns:
            Dictionary with AI skill correlation analysis
        """
        try:
            logger.info(f"🤖 Analyzing {title_1} vs {title_2} skill correlations...")
            
            title_1_jobs = df[df['role_category'] == title_1]
            title_2_jobs = df[df['role_category'] == title_2]
            
            correlation_analysis = {}
            category_cols = [col for col in df.columns if col.startswith('has_')]
            
            for category in category_cols:
                category_name = category.replace('has_', '').replace('_', ' ')
                
                title_1_percentage = (title_1_jobs[category].sum() / len(title_1_jobs)) * 100 if len(title_1_jobs) > 0 else 0
                title_2_percentage = (title_2_jobs[category].sum() / len(title_2_jobs)) * 100 if len(title_2_jobs) > 0 else 0
                
                correlation_analysis[category_name] = {
                    'title_1_jobs_percentage': title_1_percentage,
                    'title_2_jobs_percentage': title_2_percentage,
                    'difference': title_1_percentage - title_2_percentage
                }
            
            return correlation_analysis, title_1, title_2
        except Exception as e:
            print('titles not provided')

    def plot_ai_skill_correlations(self, correlation_analysis: Dict, figsize: Tuple = (12, 6)):
        """
        Visualize skill requirements for two job titles
        
        Args:
            correlation_analysis: Dictionary from analyze_skills_correlation_between_titles
            figsize: Figure size
        """
        try:
            categories, title_1, title_2 = list(correlation_analysis.keys())
            title_1_percentages = [data['title_1_percentage'] for data in correlation_analysis.values()]
            title_2_percentages = [data['title_2_jobs_percentage'] for data in correlation_analysis.values()]
            
            x = np.arange(len(categories))
            width = 0.35
            
            plt.figure(figsize=figsize)
            plt.bar(x - width/2, title_1_percentages, width, label=f'{title_1} Jobs', color='#FF6B6B')
            plt.bar(x + width/2, title_2_percentages, width, label=f'{title_2} Jobs', color='#4ECDC4')
            
            plt.xlabel('Skill Categories')
            plt.ylabel('Percentage of Jobs (%)')
            plt.title(f'Skill Requirements: {title_1} vs {title_2} Jobs')
            plt.xticks(x, categories, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(self.figures_dir / f'{title_1}_{title_2}_skill_correlations.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"✅ {title_1} vs {title_2} skill correlations visualization saved")
        
        except Exception as e:
            print("Can't generate plot due to the Exception: ",e)
    
    def analyze_temporal_trends(self, df: pd.DataFrame) -> Dict:
        """
        Analyze how skill requirements change over time
        
        Args:
            df: DataFrame with published_dt and skills_parsed columns
            
        Returns:
            Dictionary with temporal trend analysis
        """
        logger.info("📅 Analyzing temporal skill trends...")
        
        if 'published_dt' not in df.columns:
            logger.warning("❌ Publication date data not available")
            return {}
        
        # Convert to datetime and extract month-year
        df['published_dt'] = pd.to_datetime(df['published_dt'])
        df['month_year'] = df['published_dt'].dt.to_period('M')
        
        temporal_analysis = {}
        time_periods = df['month_year'].dropna().unique()
        
        for period in sorted(time_periods):
            period_data = df[df['month_year'] == period]
            
            # Flatten skills for this period
            all_skills = []
            for skills_list in period_data['skills_parsed'].dropna():
                if isinstance(skills_list, str):
                    try:
                        skills_list = eval(skills_list)
                    except:
                        continue
                all_skills.extend(skills_list)
            
            skill_counts = Counter(all_skills)
            top_skills = skill_counts.most_common(5)
            
            temporal_analysis[str(period)] = {
                'job_count': len(period_data),
                'top_skills': top_skills,
                'total_skills_mentioned': len(all_skills)
            }
        
        return temporal_analysis
    
    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive analysis report
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("📊 Generating comprehensive analysis report...")
        df['published_dt'] = pd.to_datetime(df['published_dt'],utc=True,errors='coerce')
        
        report = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'total_jobs_analyzed': len(df),
                'analysis_period': f"{df['published_dt'].min()} to {df['published_dt'].max()}" 
                if 'published_dt' in df.columns else 'Unknown'
            },
            'skill_frequency': self.analyze_skill_frequency(df),
            'skill_categories': self.analyze_skill_categories(df),
            'skill_combinations': self.analyze_skill_combinations(df),
            'seniority_analysis': self.analyze_skills_by_seniority(df),
            'location_analysis': self.analyze_skills_by_location(df),
            'title_correlations': self.analyze_skills_correlation_between_titles(df),
            'temporal_trends': self.analyze_temporal_trends(df)
        }
        
        return report
    
    def save_analysis_results(self, report: Dict, filename: str = "skills_analysis_report.json"):
        """
        Save analysis results to JSON file
        """
        output_path = self.reports_dir / filename
         
        # Convert numpy types in the report
        def convert_types(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        serializable_report = convert_types(report)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"💾 Analysis report saved to: {output_path}")
        
    def create_all_visualizations(self, df: pd.DataFrame, report: Dict):
        """
        Create all visualizations for the analysis
        
        Args:
            df: Cleaned DataFrame
            report: Analysis report dictionary
        """
        logger.info("🎨 Creating visualizations...")
        
        # Skill frequency visualization
        self.plot_top_skills(report['skill_frequency'])
        
        # Skill categories visualization
        self.plot_skill_categories(report['skill_categories'])
        
        # AI correlations visualization
        self.plot_ai_skill_correlations(report['title_correlations'])
        
        logger.info("✅ All visualizations created and saved")
    
    def run_full_analysis(self) -> Dict:
        """
        Execute the complete analysis pipeline
        
        Returns:
            Comprehensive analysis report
        """
        logger.info("🚀 Starting comprehensive skills analysis...")
        
        # Load data
        df = self.load_cleaned_data()
        
        # Generate report
        report = self.generate_comprehensive_report(df)
        
        # Create visualizations
        self.create_all_visualizations(df, report)
        
        # Save results
        self.save_analysis_results(report)
        
        logger.info("🎉 Analysis completed successfully!")
        return report


# Convenience function for quick usage
def analyze_jobs_skills(data_dir: str = "../data") -> Dict:
    """
    Convenience function to run the full analysis pipeline
    
    Args:
        data_dir: Root data directory
        
    Returns:
        Analysis report
    """
    analyzer = DataScienceJobsAnalyzer(data_dir)
    report = analyzer.run_full_analysis()
    return report


if __name__ == "__main__":
    # Example usage when run as script
    report = analyze_jobs_skills()
    print(f"Analysis complete! Report generated with {report['metadata']['total_jobs_analyzed']} jobs analyzed.")