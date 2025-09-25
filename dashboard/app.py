# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import sys
from typing import Dict
from datetime import datetime
 
# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import existing analysis modules
from scr.visualization.dashboard import DataScienceJobsDashboard
from scr.analysis.analyze_skills import DataScienceJobsAnalyzer
from scr.processing.clean_skills import DataScienceJobsCleaner

st.set_option("client.showErrorDetails", True)
# Configure the page
st.set_page_config(
    page_title="Data Science Jobs Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedDataScienceJobsDashboard(DataScienceJobsDashboard):
    """
    Enhanced dashboard with additional features and improved UI
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced dashboard"""
        super().__init__(data_dir)
        self.analyzer = DataScienceJobsAnalyzer(data_dir)
    
    def render_enhanced_dashboard(self):
        """Render the enhanced dashboard with additional features"""
        
        # Header section
        st.markdown('<h1 class="main-header">📈 Data Science Job Market Analytics</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        **Interactive dashboard for analyzing data science job market trends, skill requirements, 
        and geographic distribution. Use the filters in the sidebar to explore the data.**
        """)
        
        # Sidebar with enhanced filters
        filters = self.setup_enhanced_sidebar()
        
        # Apply filters and get filtered data
        filtered_df = self._apply_filters(filters)
        
        # Show filter summary
        self._render_filter_summary(filters, filtered_df)
        
        # Summary metrics with enhanced design
        self._render_enhanced_metrics(filtered_df)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Skills Analysis", 
            "🌍 Geographic Insights", 
            "📅 Temporal Trends", 
            "🎯 Role Comparison", 
            "📈 Advanced Analytics"
        ])
        
        with tab1:
            self._render_enhanced_skills_analysis(filtered_df)
        
        with tab2:
            self._render_enhanced_geographic_analysis(filtered_df)
        
        with tab3:
            self._render_enhanced_temporal_analysis(filtered_df)
        
        with tab4:
            self._render_enhanced_role_comparison(filtered_df)
        
        with tab5:
            self._render_advanced_analytics(filtered_df)
        
        # Footer
        self._render_footer()
    
    def setup_enhanced_sidebar(self):
        """Create enhanced sidebar with additional filters and info"""
        with st.sidebar:
            st.title("🔧 Dashboard Controls")
            st.markdown("---")
            
            # Data info card
            st.subheader("📁 Dataset Info")
            st.info(f"""
            **Total Jobs:** {len(self.df):,}
            **Companies:** {self.df['company'].nunique():,}
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
            """)
            
            st.markdown("---")
            
            # Enhanced filters
            filters = {}
            
            # Role category filter with search
            if 'role_category' in self.df.columns:
                roles = ['All'] + sorted(self.df['role_category'].dropna().unique().tolist())
                filters['role'] = st.selectbox(
                    "🎯 Filter by Role",
                    roles,
                    index=0,
                    help="Filter jobs by role category"
                )
            
            # Location filter with multi-select option
            if 'location_category' in self.df.columns:
                locations = ['All'] + sorted(self.df['location_category'].dropna().unique().tolist())
                filters['location'] = st.selectbox(
                    "🌍 Filter by Location",
                    locations,
                    index=0,
                    help="Filter jobs by location category"
                )
            
            # Enhanced AI filter with description
            if 'ai' in self.df.columns:
                st.subheader("🤖 AI Focus")
                ai_option = st.radio(
                    "AI Job Filter",
                    ['All Jobs', 'AI Jobs Only', 'Non-AI Jobs Only'],
                    help="Filter for AI-focused positions"
                )
                filters['ai_filter'] = ai_option.replace(' Jobs', '')
            
            # Skills filter with search and categories
            if 'skills_parsed' in self.df.columns:
                st.subheader("🛠️ Skills Filter")
                
                # Skill category quick select
                category_cols = [col for col in self.df.columns if col.startswith('has_')]
                if category_cols:
                    categories = [col.replace('has_', '').replace('_', ' ').title() 
                                for col in category_cols]
                    selected_categories = st.multiselect(
                        "Skill Categories",
                        categories,
                        help="Filter by skill categories"
                    )
                    filters['skill_categories'] = selected_categories
                
                # Individual skills search
                all_skills = self._extract_unique_skills()
                selected_skills = st.multiselect(
                    "Specific Skills",
                    all_skills,
                    help="Select specific skills to filter by"
                )
                filters['skills'] = selected_skills
            
            # Date range filter with presets
            if 'published_dt' in self.df.columns:
                st.subheader("📅 Time Period")
                min_date = pd.to_datetime(self.df['published_dt']).min()
                max_date = pd.to_datetime(self.df['published_dt']).max()
                
                # Quick date presets
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Last 30 Days"):
                        filters['date_preset'] = '30d'
                with col2:
                    if st.button("Last 90 Days"):
                        filters['date_preset'] = '90d'
                
                date_range = st.date_input(
                    "Custom Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                filters['date_range'] = date_range
            
            # Analysis settings
            st.markdown("---")
            st.subheader("⚙️ Analysis Settings")
            filters['top_n'] = st.slider("Number of Top Items", 5, 25, 10)
            filters['chart_style'] = st.selectbox("Chart Style", ["Plotly", "Matplotlib"])
            
            return filters
    
    def _render_filter_summary(self, filters: Dict, filtered_df: pd.DataFrame):
        """Show current filter summary"""
        if len(filtered_df) < len(self.df):
            reduction_pct = ((len(self.df) - len(filtered_df)) / len(self.df)) * 100
            st.success(f"""
            **Filters Applied:** Showing {len(filtered_df):,} of {len(self.df):,} jobs 
            ({reduction_pct:.1f}% reduction)
            """)
        else:
            st.info("**No filters applied:** Showing all available jobs")
    
    def _render_enhanced_metrics(self, df: pd.DataFrame):
        """Render enhanced metrics cards"""
        st.markdown("---")
        
        # Create columns for metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Jobs", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            companies = df['company'].nunique()
            st.metric("Companies", f"{companies:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_skills = df['skills_count'].mean() if 'skills_count' in df.columns else 0
            st.metric("Avg Skills/Job", f"{avg_skills:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            ai_pct = (df['ai'].sum() / len(df) * 100) if 'ai' in df.columns else 0
            st.metric("AI Jobs", f"{ai_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'salary' in df.columns:
                avg_salary = df['salary_min'].mean()
                st.metric("Avg Salary", f"${avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A")
            else:
                st.metric("Data Freshness", f"{(df['days_since_publication'].mean()):.0f} days")
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_enhanced_skills_analysis(self, df: pd.DataFrame):
        """Enhanced skills analysis with interactive features"""
        st.header("🛠️ Skills Analysis")
        
        if df.empty:
            st.warning("No data available with current filters")
            return
        
        # Quick analysis button
        if st.button("🔄 Update Skills Analysis", help="Re-analyze skills with current filters"):
            with st.spinner("Analyzing skills..."):
                skill_analysis = self.analyzer.analyze_skill_frequency(df)
        else:
            # Use cached analysis for performance
            skill_analysis = self.analyzer.analyze_skill_frequency(df)
        
        # Top skills in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive skills table
            st.subheader("📋 Skills Ranking")
            top_skills = skill_analysis['top_skills'][:15]
            skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Mentions'])
            skills_df['Prevalence'] = skills_df['Skill'].apply(
                lambda x: skill_analysis['skill_prevalence'].get(x, 0)
            )
            skills_df['Prevalence'] = skills_df['Prevalence'].round(1)
            skills_df['Rank'] = range(1, len(skills_df) + 1)
            
            # Display interactive table
            st.dataframe(
                skills_df[['Rank', 'Skill', 'Mentions', 'Prevalence']],
                use_container_width=True,
                height=400
            )
        
        with col2:
            # Skills distribution chart
            st.subheader("📊 Skills Distribution")
            fig = px.pie(
                values=skills_df['Mentions'],
                names=skills_df['Skill'],
                title="Skills Mention Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed charts
        col3, col4 = st.columns(2)
        
        with col3:
            # Mentions vs Prevalence scatter plot
            st.subheader("🎯 Skill Importance Matrix")
            all_skills = list(skill_analysis['skill_counts'].keys())
            mentions = [skill_analysis['skill_counts'][skill] for skill in all_skills]
            prevalence = [skill_analysis['skill_prevalence'].get(skill, 0) for skill in all_skills]

            
            fig = px.scatter(
                x=prevalence, y=mentions,
                size=[m*p for m, p in zip(mentions, prevalence)],
                hover_name=all_skills,
                title="Mentions vs Prevalence",
                labels={'x': 'Prevalence (% of Jobs)', 'y': 'Mentions'},
                size_max=30,
                color=mentions,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Skill categories analysis
            st.subheader("📚 Skill Categories")
            category_analysis = self.analyzer.analyze_skill_categories(df)
            
            if category_analysis:
                categories = list(category_analysis.keys())
                percentages = [cat_data['percentage'] for cat_data in category_analysis.values()]
                
                fig = px.bar(
                    x=percentages, y=categories,
                    orientation='h',
                    title="Skill Categories Prevalence",
                    labels={'x': 'Percentage of Jobs (%)', 'y': 'Category'},
                    color=percentages,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_enhanced_geographic_analysis(self, df: pd.DataFrame):
        """Enhanced geographic analysis"""
        st.header("🌍 Geographic Analysis")
        
        if df.empty or 'country' not in df.columns:
            st.warning("Geographic data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Location distribution
            location_counts = df['country'].value_counts()[df['country'].value_counts()>90]
            
            fig = px.pie(
                values=location_counts.values,
                names=location_counts.index,
                title="Job Distribution by Location",
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Location bar chart
            fig = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Jobs by Location",
                labels={'x': 'Number of Jobs', 'y': 'Location'},
                color=location_counts.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Location skills analysis
        st.subheader("📍 Skills by Location")
        location_analysis = self.analyzer.analyze_skills_by_location(df)
        
        if location_analysis:
            # Create a select box for location
            locations = list(location_analysis.keys())
            selected_location = st.selectbox("Select Location", locations)
            
            if selected_location:
                location_data = location_analysis[selected_location]
                top_skills = location_data['top_skills'][:10]
                
                if top_skills:
                    skills, counts = zip(*top_skills)
                    fig = px.bar(
                        x=counts, y=skills,
                        orientation='h',
                        title=f"Top Skills in {selected_location}",
                        labels={'x': 'Mentions', 'y': 'Skills'},
                        color=counts
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_enhanced_temporal_analysis(self, df: pd.DataFrame):
        """Enhanced temporal analysis"""
        st.header("📅 Temporal Trends")
        
        if df.empty or 'published_dt' not in df.columns:
            st.warning("Temporal data not available")
            return
        
        df_temp = df.copy()
        df_temp['published_dt'] = pd.to_datetime(df_temp['published_dt'])
        df_temp['month'] = df_temp['published_dt'].dt.to_period('M').astype(str)
        df_temp['week'] = df_temp['published_dt'].dt.isocalendar().week
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trends
            monthly_counts = df_temp['month'].value_counts().sort_index()
            
            fig = px.line(
                x=monthly_counts.index,
                y=monthly_counts.values,
                title="Monthly Job Postings",
                labels={'x': 'Month', 'y': 'Number of Jobs'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekly trends
            weekly_counts = df_temp.groupby('week').size()
            
            fig = px.line(
                x=weekly_counts.index,
                y=weekly_counts.values,
                title="Weekly Job Postings",
                labels={'x': 'Week', 'y': 'Number of Jobs'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_enhanced_role_comparison(self, df: pd.DataFrame):
        """Enhanced role comparison"""
        st.header("🎯 Role Comparison")
        
        if df.empty or 'role_category' not in df.columns:
            st.warning("Role data not available")
            return
        
        roles = df['role_category'].value_counts().index.tolist()
        
        if len(roles) < 2:
            st.warning("Need at least 2 roles for comparison")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            role1 = st.selectbox("Select First Role", roles, key="role1")
        with col2:
            # Filter available roles for role2
            available_roles = [r for r in roles if r != role1]
            role2 = st.selectbox("Select Second Role", available_roles, key="role2")
        
        if role1 and role2:
            try:
                correlation_analysis, title1, title2 = self.analyzer.analyze_skills_correlation_between_titles(
                    df, role1, role2
                )
                
                categories = list(correlation_analysis.keys())
                role1_pct = [correlation_analysis[cat]['title_1_jobs_percentage'] for cat in categories]
                role2_pct = [correlation_analysis[cat]['title_2_jobs_percentage'] for cat in categories]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name=role1, x=categories, y=role1_pct))
                fig.add_trace(go.Bar(name=role2, x=categories, y=role2_pct))
                
                fig.update_layout(
                    title=f"Skill Requirements: {role1} vs {role2}",
                    xaxis_title="Skill Categories",
                    yaxis_title="Percentage of Jobs (%)",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating comparison: {e}")
    
    def _render_advanced_analytics(self, df: pd.DataFrame):
        """Advanced analytics section"""
        st.header("📈 Advanced Analytics")
        
        if df.empty:
            return
        
        # Skill combinations analysis
        st.subheader("🔄 Skill Combinations")
        
        if st.button("Analyze Skill Combinations"):
            with st.spinner("Analyzing skill combinations..."):
                combo_analysis = self.analyzer.analyze_skill_combinations(df)
                
                if combo_analysis['top_combinations']:
                    top_combos = combo_analysis['top_combinations'][:10]
                    
                    combo_data = []
                    for (skill1, skill2), count in top_combos:
                        combo_data.append({
                            'Skill 1': skill1,
                            'Skill 2': skill2,
                            'Co-occurrence': count
                        })
                    
                    combo_df = pd.DataFrame(combo_data)
                    st.dataframe(combo_df, use_container_width=True)
        
        # Salary analysis if available
        if 'salary' in df.columns and not df['salary'].isna().all():
            st.subheader("💰 Salary Analysis")
            
            salary_data = df[df['salary'].notna()]
            if not salary_data.empty:
                fig = px.histogram(
                    salary_data, 
                    x='salary',
                    title="Salary Distribution",
                    labels={'salary': 'Salary'},
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_footer(self):
        """Render dashboard footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Data Science Job Market Dashboard • Built with Streamlit • 
            <a href='https://github.com/your-repo' target='_blank'>View Source Code</a></p>
        </div>
        """, unsafe_allow_html=True)

import traceback

def main():
    """Main function to run the enhanced dashboard"""
    
    # Optional: show full Streamlit error details in dev mode
    st.set_option("client.showErrorDetails", True)
    
    try:
        # Initialize dashboard
        dashboard = EnhancedDataScienceJobsDashboard()
        
        # Render the dashboard
        dashboard.render_enhanced_dashboard()
        
    except Exception as e:
        # Show a concise error message
        st.error(f"⚠️ Error initializing dashboard: {e}")
        
        # Show full traceback for debugging
        st.subheader("Debug Info:")
        st.text(traceback.format_exc())
        
        # Friendly troubleshooting tips
        st.info("""
        **Troubleshooting tips:**
        1. Ensure the data files exist in the data/interim/ directory
        2. Run the cleaning pipeline (02_cleaning.ipynb) first
        3. Check that all required packages are installed
        """)

if __name__ == "__main__":
    main()
