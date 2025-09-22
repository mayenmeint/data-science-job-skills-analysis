"""
Visualization functions for job skills analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

def setup_visualization_style():
    """
    Set up consistent visualization style for matplotlib plots.
    """
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

def plot_skill_frequency(skill_freq_df: pd.DataFrame, top_n: int = 15, 
                        title: str = "Top Skills Frequency", save_path: Optional[str] = None):
    """
    Plot horizontal bar chart of top skills by frequency.
    
    Parameters:
    skill_freq_df (DataFrame): DataFrame from analyze_skill_frequency()
    top_n (int): Number of top skills to display
    title (str): Plot title
    save_path (str): Optional path to save the figure
    """
    setup_visualization_style()
    
    top_skills = skill_freq_df.head(top_n)
    
    plt.figure(figsize=(14, 10))
    bars = plt.barh(top_skills['skill'], top_skills['frequency'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_skills))))
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + max(top_skills['frequency']) * 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.xlabel('Number of Job Postings', fontweight='bold')
    plt.ylabel('Skill', fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest frequency at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skills_by_title_heatmap(skills_by_title_df: pd.DataFrame, 
                               top_n_titles: int = 8, 
                               top_n_skills: int = 12,
                               normalize: bool = True,
                               save_path: Optional[str] = None):
    """
    Create heatmap of skills by job title.
    
    Parameters:
    skills_by_title_df (DataFrame): DataFrame from analyze_skills_by_title()
    top_n_titles (int): Number of top titles to include
    top_n_skills (int): Number of top skills to include
    normalize (bool): Whether to normalize by title frequency
    save_path (str): Optional path to save the figure
    """
    setup_visualization_style()
    
    # Get top titles by job count
    top_titles = skills_by_title_df.nlargest(top_n_titles, 'total_jobs').index
    
    # Get top skills across all titles
    skill_columns = [col for col in skills_by_title_df.columns if col != 'total_jobs']
    top_skills = skills_by_title_df[skill_columns].sum().nlargest(top_n_skills).index
    
    # Prepare data for heatmap
    heatmap_data = skills_by_title_df.loc[top_titles, top_skills]
    
    if normalize:
        # Normalize by number of jobs per title
        heatmap_data = heatmap_data.div(skills_by_title_df.loc[top_titles, 'total_jobs'], axis=0)
        vmin, vmax = 0, 1
        fmt = '.2f'
        title_suffix = ' (Normalized by Title Frequency)'
    else:
        vmin, vmax = 0, heatmap_data.max().max()
        fmt = 'd'
        title_suffix = ' (Raw Counts)'
    
    plt.figure(figsize=(16, 10))
    
    # Create heatmap
    cmap = LinearSegmentedColormap.from_list('skill_cmap', ['#f7fbff', '#08306b'])
    im = plt.imshow(heatmap_data.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set labels
    plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45, ha='right')
    plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Skill Importance' if normalize else 'Skill Count')
    
    # Add values as text
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if value > (vmax * 0.1):  # Only show text for significant values
                plt.text(j, i, f'{value:{fmt}}', ha='center', va='center', 
                       fontsize=9, fontweight='bold', 
                       color='white' if value > (vmax * 0.5) else 'black')
    
    plt.title(f'Skills by Job Title{title_suffix}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_title_distribution(title_stats_df: pd.DataFrame, 
                          top_n: int = 10,
                          save_path: Optional[str] = None):
    """
    Plot distribution of job titles.
    
    Parameters:
    title_stats_df (DataFrame): DataFrame from get_title_statistics()
    top_n (int): Number of top titles to show
    save_path (str): Optional path to save the figure
    """
    setup_visualization_style()
    
    top_titles = title_stats_df.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(top_titles['job_count'], 
                                      labels=top_titles['title'], 
                                      autopct='%1.1f%%', startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Job Title Distribution', fontweight='bold')
    
    # Bar chart of skills per title
    bars = ax2.barh(top_titles['title'], top_titles['avg_skills_per_job'],
                   color=plt.cm.plasma(np.linspace(0, 1, len(top_titles))))
    ax2.set_xlabel('Average Skills per Job')
    ax2.set_title('Average Skills by Title', fontweight='bold')
    ax2.invert_yaxis()
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_skill_associations(associations_df: pd.DataFrame, 
                          target_skill: str = 'python',
                          top_n: int = 15,
                          save_path: Optional[str] = None):
    """
    Plot association analysis for a target skill.
    
    Parameters:
    associations_df (DataFrame): DataFrame from analyze_skill_associations()
    target_skill (str): The target skill being analyzed
    top_n (int): Number of top associations to show
    save_path (str): Optional path to save the figure
    """
    setup_visualization_style()
    
    top_associations = associations_df.head(top_n)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle(f'Skill Associations with {target_skill.title()}', fontsize=16, fontweight='bold')
    
    # Co-occurrence count
    bars1 = ax1.barh(top_associations['skill'], top_associations['cooccurrence_count'])
    ax1.set_xlabel('Co-occurrence Count')
    ax1.set_title('Raw Co-occurrence')
    ax1.invert_yaxis()
    
    # Association ratio
    bars2 = ax2.barh(top_associations['skill'], top_associations['association_ratio'])
    ax2.set_xlabel('Association Ratio')
    ax2.set_title('Association Ratio (Co-occurrence / Target Frequency)')
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    
    # Lift score
    bars3 = ax3.barh(top_associations['skill'], top_associations['lift_score'])
    ax3.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Random chance (Lift=1)')
    ax3.set_xlabel('Lift Score')
    ax3.set_title('Lift Score (>1 = Positive Association)')
    ax3.legend()
    ax3.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, 
                           top_n_skills: int = 15,
                           save_path: Optional[str] = None):
    """
    Plot correlation heatmap for skills.
    
    Parameters:
    corr_matrix (DataFrame): Correlation matrix from calculate_statistical_correlations()
    top_n_skills (int): Number of top skills to include
    save_path (str): Optional path to save the figure
    """
    setup_visualization_style()
    
    # Get top skills by variance or mean correlation
    top_skills = corr_matrix.sum().nlargest(top_n_skills).index
    
    # Filter correlation matrix
    corr_top = corr_matrix.loc[top_skills, top_skills]
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_top, dtype=bool))
    
    plt.figure(figsize=(14, 12))
    
    # Create custom colormap for correlations
    cmap = LinearSegmentedColormap.from_list('corr_cmap', ['#D53E4F', 'white', '#3288BD'])
    
    # Plot heatmap
    sns.heatmap(corr_top, 
                mask=mask,
                cmap=cmap,
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                square=True,
                linewidths=0.5)
    
    plt.title(f'Skill Correlation Matrix (Top {top_n_skills} Skills)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_skill_plot(skill_freq_df: pd.DataFrame, top_n: int = 20):
    """
    Create interactive plot of skill frequencies using Plotly.
    
    Parameters:
    skill_freq_df (DataFrame): DataFrame from analyze_skill_frequency()
    top_n (int): Number of top skills to display
    """
    top_skills = skill_freq_df.head(top_n)
    
    fig = px.bar(top_skills, 
                 x='frequency', 
                 y='skill',
                 orientation='h',
                 title=f'Top {top_n} Most In-Demand Skills',
                 labels={'frequency': 'Number of Job Postings', 'skill': 'Skill'},
                 color='frequency',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    fig.show()

def plot_skill_network(cooccurrence_matrix: pd.DataFrame, 
                     min_cooccurrence: int = 3,
                     save_path: Optional[str] = None):
    """
    Create network visualization of skill co-occurrences.
    
    Parameters:
    cooccurrence_matrix (DataFrame): Co-occurrence matrix from create_skill_cooccurrence_matrix()
    min_cooccurrence (int): Minimum co-occurrence count to show connection
    save_path (str): Optional path to save the figure
    """
    try:
        import networkx as nx
        
        setup_visualization_style()
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for skill in cooccurrence_matrix.index:
            G.add_node(skill)
        
        # Add edges for significant co-occurrences
        for skill1 in cooccurrence_matrix.index:
            for skill2 in cooccurrence_matrix.columns:
                if skill1 != skill2:
                    weight = cooccurrence_matrix.loc[skill1, skill2]
                    if weight >= min_cooccurrence:
                        G.add_edge(skill1, skill2, weight=weight)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        plt.figure(figsize=(16, 12))
        
        # Draw nodes with size based on degree
        node_sizes = [G.degree(node) * 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        
        # Draw edges with width based on co-occurrence strength
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w/2 for w in edge_weights], 
                              alpha=0.6, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f'Skill Relationship Network\n(Connections show ≥{min_cooccurrence} co-occurrences)', 
                  fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("NetworkX not installed. Install with: pip install networkx")

def create_comprehensive_dashboard(df: pd.DataFrame, 
                                 title_column: str = 'title_category',
                                 skills_column: str = 'skills',
                                 save_dir: Optional[str] = None):
    """
    Create a comprehensive dashboard of all visualizations.
    
    Parameters:
    df (DataFrame): Input DataFrame
    title_column (str): Column name for job titles
    skills_column (str): Column name for skills
    save_dir (str): Optional directory to save figures
    """
    from scr.skill_analysis import analyze_skill_frequency, analyze_skills_by_title, get_title_statistics
    from scr.correlation_analysis import create_skill_cooccurrence_matrix, analyze_skill_associations, calculate_statistical_correlations
    
    print("Creating comprehensive skills analysis dashboard...")
    
    # Generate all analysis data
    skill_freq = analyze_skill_frequency(df, skills_column)
    skills_by_title = analyze_skills_by_title(df, title_column, skills_column)
    title_stats = get_title_statistics(df, title_column, skills_column)
    cooccurrence_matrix = create_skill_cooccurrence_matrix(df, skills_column)
    python_associations = analyze_skill_associations(df, 'python', skills_column)
    corr_matrix, _ = calculate_statistical_correlations(df, skills_column)
    
    # Create all visualizations
    plot_skill_frequency(skill_freq, top_n=15, 
                        title="Top Skills Across All Job Postings",
                        save_path=f"{save_dir}/skill_frequency.png" if save_dir else None)
    
    plot_skills_by_title_heatmap(skills_by_title, top_n_titles=8, top_n_skills=12,
                                normalize=True,
                                save_path=f"{save_dir}/skills_by_title_heatmap.png" if save_dir else None)
    
    plot_title_distribution(title_stats, top_n=10,
                           save_path=f"{save_dir}/title_distribution.png" if save_dir else None)
    
    plot_skill_associations(python_associations, 'python', top_n=15,
                           save_path=f"{save_dir}/python_associations.png" if save_dir else None)
    
    plot_correlation_heatmap(corr_matrix, top_n_skills=15,
                            save_path=f"{save_dir}/correlation_heatmap.png" if save_dir else None)
    
    plot_skill_network(cooccurrence_matrix, min_cooccurrence=3,
                      save_path=f"{save_dir}/skill_network.png" if save_dir else None)
    
    print("Dashboard creation complete!")
