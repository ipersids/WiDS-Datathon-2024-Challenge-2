<center> <h1> WiDS Datathon 2024 Challenge #2<h1> </center>
[Datathon 2024 Challenge Theme: Equity in Healthcare.](https://www.widsworldwide.org/wp-content/uploads/2023/09/Datathon-Oncology-HomePage-16x9-1.png)

[The WiDS Datathon](https://www.widsworldwide.org/learn/datathon/) is hosted on [Kaggle](https://www.kaggle.com/competitions/widsdatathon2024-challenge2), using a predictive analytics challenge focused on social impact. This Datathon used a real-world evidence dataset from Health Verity, one of the largest healthcare data ecosystems in the US.

**Data description:**.  
The datasets contain health-related information on patients diagnosed with metastatic triple-negative breast cancer in the USA. In addition, social, economic, demographic and climatic information was included using zip codes.

**Task:**  
Predict the time of metastatic diagnosis for patients in the test dataset using the patient characteristics and information provided.

**Libraries:**  
```python
# Data manipulation
import pandas as pd
import numpy as np
import re
import datetime as dt

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Mathematical and statistical packages
import math import ceil
from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr

#Machine Learning Libraries
# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
# Clustering
from sklearn.cluster import KMeans
# Linear models
from sklearn.linear_model import Ridge
# Tree-based models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
# Neural Networks
from sklearn.neural_network import MLPRegressor
# Model performance metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score, f1_score


# modules and libraries for NLP processing the text data
import string
import nltk
```

## Work Process
- Exploratory Data Analysis (EDA)
    - Key takeaways from the discovery phase
    - Key takeaways from the structuring phase
    - Cleaning data strategy overview
    - Resulting dataset validation
- Attribute Selection and Final Data Transformation
- Machine Learning Phase
    - Construction phase
    - Execution phase
    - Submission
<br>
</br>

**Submission:**  
The final prediction was made using the *Gradient Boosting Regressor* model, which performed best against other models in terms of Root Mean Squared Error (RMSE).
<br>

</br>

____
**Team of [DataTribe Collective](https://www.linkedin.com/company/datatribe-collective/):**  
- Julia Persidskaia ([LinkedIn](https://www.linkedin.com/in/iuliia-persidskaia/))
- Irem Corum Aktas ([LinkedIn](https://www.linkedin.com/in/irem-corum-aktas-618367b2/))
- Eevamaija Virtanen ([LinkedIn](https://www.linkedin.com/in/eevamaijavirtanen/))
