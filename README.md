# 🚗 Car Price Predictor

A machine learning project that predicts used car selling prices using multiple regression algorithms with LLM-enhanced analysis.

## 📺 Project Demo

<!-- Embed your video here -->
<div align="center">
  <a href="https://drive.google.com/file/d/188efBbqa9rB_Aqpl4t3GM8RtEGcgQhCk/view?usp=sharing">
    <img src="https://img.shields.io/badge/Watch-Demo%20Video-red?style=for-the-badge&logo=youtube" alt="Demo Video" />
  </a>
</div>

*Replace `YOUR_VIDEO_LINK_HERE` with your actual video link*

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline to predict used car selling prices based on various vehicle characteristics. The project leverages the power of Large Language Models (LLMs) to enhance data analysis and feature engineering decisions throughout the development process.

### Key Features
- **Multi-model comparison**: Linear Regression, Random Forest, Decision Tree, and KNN
- **LLM-enhanced analysis**: Used Gemini 2.0 Flash for data insights and cleaning recommendations
- **Feature engineering**: Created new features like car age for better predictions
- **High accuracy**: Achieved R² scores up to 0.9977 on sample predictions

## 📊 Dataset

The project uses a comprehensive car dataset with the following features:
- **name**: Car model name
- **year**: Manufacturing year
- **selling_price**: Target variable (price in currency units)
- **km_driven**: Total kilometers driven
- **fuel**: Fuel type (Petrol, Diesel, CNG, LPG, Electric)
- **seller_type**: Type of seller
- **transmission**: Manual or Automatic
- **owner**: Number of previous owners
- **mileage**: Fuel efficiency
- **engine**: Engine capacity
- **max_power**: Maximum power output
- **seats**: Number of seats

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms
- **matplotlib & seaborn** - Data visualization
- **LangChain** - LLM integration
- **Google Gemini 2.0 Flash** - Large Language Model for enhanced analysis

## 📈 Model Performance

| Model | R² Score | MSE |
|-------|----------|-----|
| **Random Forest** | **0.9656** | **0.0372** |
| Decision Tree | 0.9543 | 0.0493 |
| Linear Regression | 0.6891 | 0.3354 |
| KNN | 0.6234 | 0.4064 |

**Random Forest Regressor** emerged as the best performing model with an R² score of **96.56%**.

## 🔍 Key Insights from LLM Analysis

The integration of Gemini 2.0 Flash provided valuable insights:

1. **Data Understanding**: LLM helped interpret column meanings and relationships
2. **Cleaning Strategies**: Recommended optimal approaches for handling missing values
3. **Feature Engineering**: Suggested creating car_age feature from manufacturing year
4. **Correlation Analysis**: Provided insights on feature relationships and redundancy
5. **Model Interpretation**: Helped understand model performance and feature importance

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn langchain-google-genai python-dotenv
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/ShishirRmc/Car-Price-predictor.git
cd Car-Price-predictor
```

2. Set up environment variables:
```bash
# Create .env file and add your Gemini API key
echo "gemini_api_key=YOUR_API_KEY_HERE" > .env
```

3. Run the Jupyter notebook:
```bash
jupyter notebook car_prediction.ipynb
```

## 📋 Project Workflow

### 1. Data Loading & Exploration
- Loaded car dataset using pandas
- Performed initial data exploration with `.info()`, `.describe()`, and `.head()`
- Identified missing values and data types

### 2. LLM-Enhanced Data Understanding
- Utilized Gemini 2.0 Flash to interpret dataset columns
- Received recommendations for data cleaning strategies
- Got insights on feature relationships and preprocessing steps

### 3. Data Preprocessing
- **Missing Value Imputation**:
  - Numerical features: Filled with median/mean
  - Converted max_power to numeric with error handling
- **Feature Engineering**:
  - Created `car_age` feature (2025 - year)
  - Applied Label Encoding to categorical variables
- **Feature Selection**:
  - Dropped redundant features based on correlation analysis
  - Removed 'year' and 'name' columns

### 4. Exploratory Data Analysis
- Created comprehensive visualizations:
  - Price distribution histogram
  - Scatter plots for price vs. key features
  - Box plots for categorical feature analysis
  - Correlation heatmap
- LLM provided interpretation of visualization insights

### 5. Model Training & Evaluation
- Trained four regression models
- Applied StandardScaler to target variable
- Used 80-20 train-test split
- Evaluated using MSE and R² metrics

### 6. Feature Importance Analysis
- Analyzed feature importance for tree-based models
- Identified key price drivers:
  - **Random Forest**: Most important features above mean importance threshold
  - **Decision Tree**: Complementary feature importance insights

### 7. Model Validation
- Tested on sample data (10 and 100 random samples)
- Achieved impressive accuracy: **R² = 0.9977** on 10-sample test

## 📊 Results & Insights

### Model Performance Summary
- **Random Forest** achieved the highest accuracy (96.56% R²)
- Significantly outperformed linear models, indicating non-linear relationships in data
- Tree-based models effectively captured complex feature interactions

### Key Price Factors
Feature importance analysis revealed the most influential factors in determining car prices:
- Engine specifications
- Car age and mileage
- Brand and model characteristics
- Fuel type and transmission

### Sample Predictions Accuracy
- **10 Random Samples**: R² = 0.9977 (99.77% accuracy)
- **100 Random Samples**: R² = 0.98 (98% accuracy)

## 🤖 LLM Enhancement Benefits

The integration of Gemini 2.0 Flash significantly enhanced the project in several ways:

1. **Faster Data Understanding**: Quickly interpreted complex dataset structure
2. **Informed Decision Making**: Data-driven recommendations for preprocessing
3. **Feature Engineering Insights**: Suggested valuable feature transformations
4. **Analysis Interpretation**: Helped understand visualization patterns and model results
5. **Quality Assurance**: Identified potential data quality issues and solutions

## 📁 Project Structure

```
Car-Price-predictor/
├── car_prediction.ipynb    # Main Jupyter notebook
├── car.csv                # Dataset (not included in repo)
├── car_price_model.pkl    # Trained model (generated)
├── .env                   # Environment variables
└── README.md             # This file
```

## 🎯 Future Improvements

- Implement cross-validation for more robust model evaluation
- Explore advanced feature engineering techniques
- Add ensemble methods for improved predictions
- Deploy model as a web application
- Incorporate more recent car data for better generalization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Shishir Rmc**
- GitHub: [@ShishirRmc](https://github.com/ShishirRmc)
- Project Link: [https://github.com/ShishirRmc/Car-Price-predictor](https://github.com/ShishirRmc/Car-Price-predictor)

## 🙏 Acknowledgments

- Dataset source contributors
- Google Gemini team for the powerful LLM capabilities
- Open-source community for the amazing libraries used

---

⭐ If you found this project helpful, please give it a star!
