"""
Interactive Tableau-like dashboard for job skills analysis using Plotly and Dash.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from skill_analysis import (
    analyze_skill_frequency, 
    analyze_skills_by_title, 
    get_top_skills_by_title,
    get_title_statistics
)
from correlation_analysis import (
    create_skill_cooccurrence_matrix,
    analyze_skill_associations
)

# Load and preprocess data
def load_data():
    """Load and preprocess the job data."""
    try:
        df = pd.read_csv('data/processed/cleaned_job_data.csv')
        
        # Ensure skills are in list format
        if isinstance(df['skills'].iloc[0], str):
            from skill_analysis import get_skill_list
            df['skills'] = df['skills'].apply(get_skill_list)
            
        return df
    except FileNotFoundError:
        print("Processed data not found. Please run the analysis notebooks first.")
        return None

# Load data
df = load_data()
title_categories = []
skill_freq = pd.DataFrame()
title_stats = pd.DataFrame()
if df is not None:
    # Precompute data for dashboard
    title_categories = sorted(df['title_category'].unique())
    skill_freq = analyze_skill_frequency(df, 'skills')
    title_stats = get_title_statistics(df, 'title_category', 'skills')
    skills_by_title = analyze_skills_by_title(df, 'title_category', 'skills')
    
    # Get top skills for each title (precompute for performance)
    top_skills_by_title = {}
    for title in title_categories:
        title_df = df[df['title_category'] == title]
        if len(title_df) > 0:
            top_skills_by_title[title] = analyze_skill_frequency(title_df, 'skills').head(10)

# Initialize Dash app
app = dash.Dash(__name__, title='Job Skills Dashboard', suppress_callback_exceptions=True)
server = app.server

# Layout
app.layout = html.Div([
    html.H1("Job Skills Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Filters
    html.Div([
        html.Div([
            html.Label("Select Job Title Category:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='title-dropdown',
                options=[{'label': title, 'value': title} for title in title_categories],
                value=title_categories[0] if title_categories else None,
                clearable=False,
                style={'width': '100%'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),
        
        html.Div([
            html.Label("Select Skill:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='skill-dropdown',
                options=[],
                clearable=True,
                style={'width': '100%'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px', 'float': 'right'})
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # KPI Cards
    html.Div([
        html.Div([
            html.H4("Total Jobs", style={'color': '#6c757d'}),
            html.H2(id='total-jobs-kpi', style={'color': '#007bff'})
        ], className='card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%', 
                                  'padding': '20px', 'backgroundColor': 'white', 
                                  'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
        
        html.Div([
            html.H4("Unique Skills", style={'color': '#6c757d'}),
            html.H2(id='unique-skills-kpi', style={'color': '#28a745'})
        ], className='card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%',
                                  'padding': '20px', 'backgroundColor': 'white',
                                  'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
        
        html.Div([
            html.H4("Avg Skills/Job", style={'color': '#6c757d'}),
            html.H2(id='avg-skills-kpi', style={'color': '#ffc107'})
        ], className='card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%',
                                  'padding': '20px', 'backgroundColor': 'white',
                                  'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'}),
        
        html.Div([
            html.H4("Top Skill", style={'color': '#6c757d'}),
            html.H2(id='top-skill-kpi', style={'color': '#dc3545'})
        ], className='card', style={'width': '23%', 'display': 'inline-block', 'margin': '1%',
                                  'padding': '20px', 'backgroundColor': 'white',
                                  'borderRadius': '10px', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
    ], style={'marginBottom': '30px'}),
    
    # Main charts
    html.Div([
        # Left column
        html.Div([
            dcc.Graph(id='title-distribution-chart'),
            dcc.Graph(id='skills-heatmap-chart')
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # Right column
        html.Div([
            dcc.Graph(id='top-skills-chart'),
            dcc.Graph(id='skill-associations-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'})
    ]),
    
    # Bottom row
    html.Div([
        dcc.Graph(id='correlation-heatmap-chart', style={'height': '600px'})
    ], style={'marginTop': '30px'}),
    
    # Last updated
    html.Div([
        html.P(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
              style={'textAlign': 'center', 'color': '#6c757d', 'marginTop': '30px'})
    ])
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})

# Callbacks
@app.callback(
    [Output('skill-dropdown', 'options'),
     Output('total-jobs-kpi', 'children'),
     Output('unique-skills-kpi', 'children'),
     Output('avg-skills-kpi', 'children'),
     Output('top-skill-kpi', 'children')],
    [Input('title-dropdown', 'value')]
)
def update_metrics(selected_title):
    """Update metrics and skill dropdown based on selected title."""
    if df is None:
        return [], 'N/A', 'N/A', 'N/A', 'N/A'
    
    
    if selected_title == 'All':
        title_df = df
        skills = skill_freq['skill'].tolist()
    else:
        title_df = df[df['title_category'] == selected_title]
        skills = top_skills_by_title.get(selected_title, pd.DataFrame())['skill'].tolist()
    
    # Calculate metrics
    total_jobs = len(title_df)
    unique_skills = len(set(skill for skills_list in title_df['skills'] for skill in skills_list))
    avg_skills = title_df['skills'].apply(len).mean() if total_jobs > 0 else 0
    
    top_skill = top_skills_by_title.get(selected_title, pd.DataFrame())['skill'].iloc[0]
    
    # Create skill options
    skill_options = [{'label': skill, 'value': skill} for skill in skills]
    
    return (skill_options, 
            f"{total_jobs:,}", 
            f"{unique_skills:,}", 
            f"{avg_skills:.1f}", 
            top_skill)

@app.callback(
    Output('title-distribution-chart', 'figure'),
    [Input('title-dropdown', 'value')]
)
def update_title_distribution(selected_title):
    """Update title distribution chart."""
    fig = px.pie(title_stats, values='job_count', names='title', 
                 title='Job Title Distribution',
                 hover_data=['avg_skills_per_job'])
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    
    return fig

@app.callback(
    Output('top-skills-chart', 'figure'),
    [Input('title-dropdown', 'value')]
)
def update_top_skills(selected_title):
    """Update top skills chart for selected title."""
    if selected_title == 'All':
        skills_data = skill_freq.head(10)
        title = 'All Titles'
    else:
        skills_data = top_skills_by_title.get(selected_title, pd.DataFrame()).head(10)
        title = selected_title
    
    if len(skills_data) == 0:
        return go.Figure().add_annotation(text="No data available", showarrow=False)
    
    fig = px.bar(skills_data, x='frequency', y='skill', orientation='h',
                 title=f'Top Skills for {title}',
                 labels={'frequency': 'Number of Jobs', 'skill': 'Skill'})
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

@app.callback(
    Output('skills-heatmap-chart', 'figure'),
    [Input('title-dropdown', 'value')]
)
def update_skills_heatmap(selected_title):
    """Update skills heatmap."""
    if selected_title == 'All':
        # Use overall skills by title
        heatmap_data = skills_by_title.drop('total_jobs', axis=1)
        # Normalize
        heatmap_data = heatmap_data.div(skills_by_title['total_jobs'], axis=0)
        title = 'Skills Heatmap (All Titles)'
    else:
        # For specific title, show skill frequencies
        title_df = df[df['title_category'] == selected_title]
        skill_counts = {}
        for skills_list in title_df['skills']:
            for skill in skills_list:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Get top skills
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        skills, counts = zip(*top_skills)
        heatmap_data = pd.DataFrame({'skill': skills, 'count': counts})
        title = f'Top Skills Heatmap - {selected_title}'
    
    fig = px.imshow(heatmap_data.values if selected_title == 'All' else [counts],
                   x=heatmap_data.columns.tolist() if selected_title == 'All' else skills,
                   y=[selected_title] if selected_title != 'All' else heatmap_data.index.tolist(),
                   title=title,
                   color_continuous_scale='Blues')
    
    return fig

@app.callback(
    Output('skill-associations-chart', 'figure'),
    [Input('skill-dropdown', 'value'),
     Input('title-dropdown', 'value')]
)
def update_skill_associations(selected_skill, selected_title):
    """Update skill associations chart."""
    if not selected_skill:
        return go.Figure().add_annotation(text="Select a skill to see associations", showarrow=False)
    
    if selected_title == 'All':
        analysis_df = df
    else:
        analysis_df = df[df['title_category'] == selected_title]
    
    associations = analyze_skill_associations(analysis_df, selected_skill, 'skills')
    
    if len(associations) == 0:
        return go.Figure().add_annotation(text="No associations found", showarrow=False)
    
    fig = px.bar(associations.head(10), x='cooccurrence_count', y='skill', orientation='h',
                 title=f'Skills Associated with {selected_skill.title()}',
                 hover_data=['association_ratio', 'lift_score'])
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

@app.callback(
    Output('correlation-heatmap-chart', 'figure'),
    [Input('title-dropdown', 'value')]
)
def update_correlation_heatmap(selected_title):
    """Update correlation heatmap."""
    if selected_title == 'All':
        analysis_df = df
        title = 'Skill Correlation Matrix (All Titles)'
    else:
        analysis_df = df[df['title_category'] == selected_title]
        title = f'Skill Correlation Matrix - {selected_title}'
    
    # Create co-occurrence matrix
    cooccurrence_matrix = create_skill_cooccurrence_matrix(analysis_df, 'skills')
    
    # Get top skills for better visualization
    top_skills = cooccurrence_matrix.sum().nlargest(15).index
    corr_data = cooccurrence_matrix.loc[top_skills, top_skills]
    
    fig = px.imshow(corr_data, 
                   x=corr_data.columns.tolist(),
                   y=corr_data.index.tolist(),
                   title=title,
                   color_continuous_scale='RdBu_r',
                   aspect='auto')
    
    fig.update_layout(height=500)
    return fig

# Run the app
if __name__ == '__main__':
    print("Starting dashboard server...")
    print("Open http://localhost:8050 in your browser")
    app.run(debug=True, host='0.0.0.0', port=8050)