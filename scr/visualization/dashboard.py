# src/visualization/dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys
from typing import Dict, List, Optional
import logging

# Add src directory to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / 'src'))

# Import analysis functions from other modules
from ..analysis.analyze_skills import DataScienceJobsAnalyzer
from ..processing.clean_skills import DataScienceJobsCleaner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataScienceJobsDashboard:
    """
    Interactive Dashboard for Data Science Job Market Analysis
    Using existing analysis functions from other modules
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the dashboard with existing analysis modules
        """
        self.data_dir = Path(data_dir)
        self.analyzer = DataScienceJobsAnalyzer(data_dir)
        self.cleaner = DataScienceJobsCleaner(data_dir)
        
        # Load data and analysis
        self.df = self.analyzer.load_cleaned_data()
        self.analysis_report = self.load_analysis_report()
        
        # Set Plotly template
        self.template = "plotly_white"
    
    def load_analysis_report(self) -> Dict:
        """Load existing analysis report"""
        try:
            report_path = self.analyzer.reports_dir / "skills_analysis_report.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}
    
    def setup_sidebar_filters(self):
        """Create interactive filters using existing data structure"""
        st.sidebar.title("🔧 Dashboard Filters")
        st.sidebar.markdown("---")
        
        filters = {}
        
        # Role category filter
        if 'role_category' in self.df.columns:
            roles = ['All'] + sorted(self.df['role_category'].dropna().unique().tolist())
            filters['role'] = st.sidebar.selectbox("🎯 Role Category", roles)
        
        # Location filter
        if 'location_category' in self.df.columns:
            locations = ['All'] + sorted(self.df['location_category'].dropna().unique().tolist())
            filters['location'] = st.sidebar.selectbox("🌍 Location", locations)
        
        # AI filter
        if 'ai' in self.df.columns:
            ai_options = ['All', 'AI Jobs Only', 'Non-AI Jobs Only']
            filters['ai_filter'] = st.sidebar.radio("🤖 AI Jobs", ai_options)
        
        # Skills multiselect
        if 'skills_parsed' in self.df.columns:
            all_skills = self._extract_unique_skills()
            filters['skills'] = st.sidebar.multiselect(
                "🛠️ Skills Filter", 
                all_skills, 
                placeholder="Select skills..."
            )
        
        return filters
    
    def _extract_unique_skills(self) -> List[str]:
        """Extract unique skills from existing data"""
        skills_set = set()
        for skills_str in self.df['skills_parsed'].dropna():
            if isinstance(skills_str, str) and skills_str.startswith('['):
                try:
                    skills = eval(skills_str)
                    skills_set.update(skills)
                except:
                    continue
        return sorted(list(skills_set))
    
    def _apply_filters(self, filters: Dict) -> pd.DataFrame:
        """Apply filters to the dataset"""
        df_filtered = self.df.copy()
        
        # Role filter
        if filters.get('role') and filters['role'] != 'All':
            df_filtered = df_filtered[df_filtered['role_category'] == filters['role']]
        
        # Location filter
        if filters.get('location') and filters['location'] != 'All':
            df_filtered = df_filtered[df_filtered['location_category'] == filters['location']]
        
        # AI filter
        if filters.get('ai_filter'):
            if filters['ai_filter'] == 'AI Jobs Only':
                df_filtered = df_filtered[df_filtered['ai'] == True]
            elif filters['ai_filter'] == 'Non-AI Jobs Only':
                df_filtered = df_filtered[df_filtered['ai'] == False]
        
        # Skills filter
        if filters.get('skills'):
            def has_skills(skills_str, target_skills):
                if pd.isna(skills_str):
                    return False
                try:
                    skills = eval(skills_str) if isinstance(skills_str, str) else skills_str
                    return any(skill in skills for skill in target_skills)
                except:
                    return False
            
            mask = df_filtered['skills_parsed'].apply(
                lambda x: has_skills(x, filters['skills'])
            )
            df_filtered = df_filtered[mask]
        
        return df_filtered
    
    def render_dashboard(self):
        """Main dashboard rendering function"""
        st.set_page_config(
            page_title="Data Science Jobs Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📈 Data Science Job Market Dashboard")
        st.markdown("Interactive analysis of data science job trends and skill requirements")
        
        # Setup filters
        filters = self.setup_sidebar_filters()
        filtered_df = self._apply_filters(filters)
        
        # Summary metrics
        self._render_summary_metrics(filtered_df)
        
        # Skills analysis section
        self._render_skills_analysis(filtered_df)
        
        # Geographic analysis
        self._render_geographic_analysis(filtered_df)
        
        # Temporal analysis
        self._render_temporal_analysis(filtered_df)
        
        # Role comparison
        self._render_role_comparison()
    
    def _render_summary_metrics(self, df: pd.DataFrame):
        """Render summary metrics cards"""
        st.header("📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Jobs", f"{len(df):,}")
        
        with col2:
            companies = df['company'].nunique()
            st.metric("Unique Companies", f"{companies:,}")
        
        with col3:
            avg_skills = df['skills_count'].mean() if 'skills_count' in df.columns else 0
            st.metric("Avg Skills/Job", f"{avg_skills:.1f}")
        
        with col4:
            ai_pct = (df['ai'].sum() / len(df) * 100) if 'ai' in df.columns else 0
            st.metric("AI Jobs", f"{ai_pct:.1f}%")
    
    def _render_skills_analysis(self, df: pd.DataFrame):
        """Render skills analysis using existing analysis functions"""
        st.header("🛠️ Skills Analysis")
        
        # Use existing analyzer to get skill frequency
        skill_analysis = self.analyzer.analyze_skill_frequency(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top skills by mentions
            top_skills = skill_analysis['top_skills'][:15]
            skills, counts = zip(*top_skills)
            
            fig = px.bar(
                x=counts, y=skills,
                orientation='h',
                title="Top Skills by Mentions",
                labels={'x': 'Number of Mentions', 'y': 'Skills'},
                color=counts,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Skill prevalence
            prevalence_data = [
                (skill, skill_analysis['skill_prevalence'][skill]) 
                for skill, _ in top_skills
            ]
            skills, prevalence = zip(*prevalence_data)
            
            fig = px.bar(
                x=prevalence, y=skills,
                orientation='h',
                title="Skill Prevalence (% of Jobs)",
                labels={'x': 'Percentage of Jobs (%)', 'y': 'Skills'},
                color=prevalence,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot: Mentions vs Prevalence
        st.subheader("Skill Importance Matrix")
        
        all_skills = list(skill_analysis['skill_counts'].keys())
        mentions = [skill_analysis['skill_counts'][skill] for skill in all_skills]
        prevalence = [skill_analysis['skill_prevalence'][skill] for skill in all_skills]
        
        fig = px.scatter(
            x=prevalence, y=mentions,
            size=[m*p for m, p in zip(mentions, prevalence)],
            hover_name=all_skills,
            title="Mentions vs Prevalence",
            labels={'x': 'Prevalence (% of Jobs)', 'y': 'Number of Mentions'},
            size_max=50,
            color=mentions,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_geographic_analysis(self, df: pd.DataFrame):
        """Render geographic distribution"""
        if 'location_category' in df.columns and not df.empty:
            st.header("🌍 Geographic Distribution")
            
            location_counts = df['location_category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=location_counts.values,
                    names=location_counts.index,
                    title="Job Distribution by Location"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=location_counts.values,
                    y=location_counts.index,
                    orientation='h',
                    title="Jobs by Location",
                    labels={'x': 'Number of Jobs', 'y': 'Location'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_temporal_analysis(self, df: pd.DataFrame):
        """Render temporal trends"""
        if 'published_dt' in df.columns and not df.empty:
            st.header("📅 Temporal Trends")
            
            df_temp = df.copy()
            df_temp['published_dt'] = pd.to_datetime(df_temp['published_dt'])
            df_temp['month'] = df_temp['published_dt'].dt.to_period('M').astype(str)
            
            monthly_counts = df_temp['month'].value_counts().sort_index()
            
            fig = px.line(
                x=monthly_counts.index,
                y=monthly_counts.values,
                title="Job Postings Over Time",
                labels={'x': 'Month', 'y': 'Number of Jobs'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_role_comparison(self):
        """Render role comparison using existing analysis"""
        if 'role_category' in self.df.columns:
            st.header("🎯 Role Comparison")
            
            roles = self.df['role_category'].value_counts().head(2).index.tolist()
            
            if len(roles) >= 2:
                # Use existing analyzer for role comparison
                correlation_analysis, title1, title2 = self.analyzer.analyze_skills_correlation_between_titles(
                    self.df, roles[0], roles[1]
                )
                
                categories = list(correlation_analysis.keys())
                title1_pct = [correlation_analysis[cat]['title_1_jobs_percentage'] for cat in categories]
                title2_pct = [correlation_analysis[cat]['title_2_jobs_percentage'] for cat in categories]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name=title1, x=categories, y=title1_pct))
                fig.add_trace(go.Bar(name=title2, x=categories, y=title2_pct))
                
                fig.update_layout(
                    title=f"Skill Requirements: {title1} vs {title2}",
                    xaxis_title="Skill Categories",
                    yaxis_title="Percentage of Jobs (%)",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the dashboard"""
    dashboard = DataScienceJobsDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()