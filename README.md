# AI & ML Internship Task 1 - Titanic Dataset Preprocessing

## Overview
This repository contains the completed Task 1 (Data Preprocessing) for the AI & ML Internship at Elevate Labs.

## Repository Structure
```
elevate-lab-internship/
├── task_1/                     # Task 1: Data Preprocessing
│   ├── preprocessing.py        # Data preprocessing script
│   ├── titanic_original.csv    # Original Titanic dataset
│   ├── titanic_processed.csv   # Processed dataset after cleaning
│   ├── outlier_boxplots.png    # Visualization of outliers
│   └── requirements.txt        # Python dependencies
└── README.md                   # This file
```

## Objective
Understand data using statistics and visualizations to gain insights into the Titanic dataset.

## Tools Used
- Python 3.x
- Pandas - Data manipulation and analysis
- NumPy - Numerical operations
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization

## Dataset
The Titanic dataset contains information about passengers aboard the Titanic, including whether they survived or not. Key features include:
- survived: Survival (0 = No, 1 = Yes)
- pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- sex: Sex of the passenger
- age: Age in years
- sibsp: Number of siblings/spouses aboard
- parch: Number of parents/children aboard
- fare: Passenger fare
- embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Task 1: Data Preprocessing

### Analysis Performed
1. **Data Cleaning**: Handled missing values in age, embarked, and deck columns
2. **Feature Engineering**: Created new features like family size, title extraction, etc.
3. **Outlier Detection**: Identified and visualized outliers in fare and age using boxplots
4. **Data Transformation**: Normalized/scaled features where necessary
5. **Encoding**: Converted categorical variables to numerical format

### Files Created
- `preprocessing.py`: Complete preprocessing pipeline
- `titanic_processed.csv`: Cleaned and processed dataset
- `outlier_boxplots.png`: Visualization showing outliers in fare and age

## Key Learnings
- Proper data cleaning is crucial for accurate analysis
- Feature engineering can reveal hidden patterns (e.g., titles correlate with survival)
- Outlier treatment affects model performance and interpretation

## How to Run

```bash
# Navigate to task_1 directory
cd task_1

# Ensure you have Python 3.x installed
# Install required packages:
pip install -r requirements.txt

# Run the preprocessing script:
python preprocessing.py

# Output: titanic_processed.csv and outlier_boxplots.png
```

## Submission
After completing the task, the GitHub repository link should be submitted via the provided submission link.

---
*This repository contains the completed work for Task 1 of the AI & ML Internship at Elevate Labs.*
