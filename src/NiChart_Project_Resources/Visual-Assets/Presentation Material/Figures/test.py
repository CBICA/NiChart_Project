from github import Github
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

# Topics to search
topics = {
    "Neuroimaging": [
        "neuroimaging", "neuroimaging-analysis", "neuroimaging-data", 
        "neuroimaging-data-science", "computational-neuroimaging", 
        "neuroimaging-experiments"
    ],
    "Harmonization": [
        "harmonization", "image-harmonization", "mri-harmonization"
    ],
    "MRI": [
        "mri", "mri-images", "diffusion-mri", "mri-reconstruction", 
        "mri-brain", "mri-segmentation", "brain-mri", "fmri", 
        "fmri-data-analysis", "fmri-preprocessing", "fmri-analysis", 
        "resting-state-fmri"
    ],
    "Imaging Biomarkers": [
        "biomarkers", "prognostic-biomarkers", "image-biomarkers", 
        "voice-biomarkers", "multimodal-biomarkers", "digital-biomarkers"
    ]
}

# Function to fetch repository data from GitHub using PyGithub
def fetch_repos(github, topic, year):
    query = f"topic:{topic} created:{year}-01-01..{year}-12-31"
    result = github.search_repositories(query=query)
    return result.totalCount

# Function to collect data for multiple topics and years
def collect_data(github, topics, years):
    data = {category: {year: 0 for year in years} for category in topics.keys()}
    
    for category, subtopics in topics.items():
        for subtopic in subtopics:
            for year in years:
                count = fetch_repos(github, subtopic, year)
                data[category][year] += count
                print(f"Category: {category}, Subtopic: {subtopic}, Year: {year}, Count: {count}")
    
    return data

# Function to save data to a file
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Function to load data from a file
def load_data(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        return None

# Function to plot cumulative data
def plot_data(data):
    years = list(next(iter(data.values())).keys())
    
    plt.figure(figsize=(14, 8))
    
    cumulative_totals = {year: 0 for year in years}
    
    for category, counts in data.items():
        cumulative_counts = []
        cumulative_sum = 0
        for year in years:
            cumulative_sum += counts[year]
            cumulative_totals[year] += counts[year]
            cumulative_counts.append(cumulative_sum)
        
        plt.plot(years, cumulative_counts, marker='o', label=category)
    
    # Plot cumulative total for all topics
    cumulative_total_counts = []
    cumulative_sum = 0
    for year in years:
        cumulative_sum += cumulative_totals[year]
        cumulative_total_counts.append(cumulative_sum)
    
    plt.plot(years, cumulative_total_counts, marker='o', label='Total', linestyle='--', linewidth=2)
    
    plt.xlabel('Year')
    plt.ylabel('Cumulative Number of Repositories')
    plt.title('Cumulative GitHub Repositories per Year for Different Topics')
    plt.legend()
    plt.grid(True)
    plt.xticks(years)  # Ensure years are shown as integers
    plt.show()

# Define the years of interest
years = list(range(datetime.now().year - 10, datetime.now().year + 1))

# Your GitHub personal access token (ensure you have permissions)
token = "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"

# Initialize PyGithub
github = Github(token)

# Filename to save data
filename = "github_repo_data.pkl"

# Load data if it exists, otherwise collect and save data
data = load_data(filename)
if data is None:
    data = collect_data(github, topics, years)
    save_data(data, filename)

# Plot the data
plot_data(data)
