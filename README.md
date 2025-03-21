# TellCo Telecom Data Analysis

## Overview

The **TellCo Telecom Data Analysis** project focuses on analyzing telecommunications data to uncover insights into user engagement, network performance, and customer satisfaction. This project aims to provide stakeholders with valuable information to make informed decisions based on comprehensive data analysis.

## Features

- **User Overview Analysis**: Identifies top handsets, manufacturers, and user behavior metrics.
- **User Engagement Analysis**: Analyzes user engagement metrics, including session frequency and total traffic, and segments users using K-means clustering.
- **Experience Analysis**: Evaluates user experience based on network parameters and device characteristics, including TCP retransmission and RTT.
- **Satisfaction Analysis**: Computes engagement and experience scores to assess customer satisfaction and predicts satisfaction using regression models.
- **Data Aggregation and Visualization**: Provides comprehensive insights through interactive visualizations.

## Technologies Used

- Python 3.9
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Docker

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (for containerization)

### Clone the Repository

```bash
git clone https://github.com/Selam46/tellco_telecom_data_analysis.git
cd telleco_telecom_data_analysis
```

### Install Dependencies

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Usage

To run the analysis scripts, execute the relevant Python files in your environment. For visualizations, you can use Jupyter notebooks or run the scripts directly.

## Testing

To run the unit tests for the project, use the following command:

```bash
python -m unittest discover -s tests
```

### Test Coverage

The project includes unit tests for the following modules:

- `data_processing.py`: Tests for data generation and aggregation functions.
- `engagement_analysis.py`: Tests for user engagement metrics and clustering functions.

## Docker Setup

To build and run the application as a Docker container, follow these steps:

### Build the Docker Image

```bash
docker build -t your-image-name .
```

### Run the Docker Container

```bash
docker run -p 8501:8501 tellco-analytics
```

### Accessing the Application in Docker

Open your web browser and navigate to `http://localhost:8501` to access the dashboard running in the Docker container.



