import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from typing import List
from .model_loader import load_rf_model,predict_overhead,calculate_rcl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from typing import List
import os
import pandas as pd


import numpy as np
import pandas as pd
import scipy.stats as stats

def analyze_distribution(data: pd.DataFrame):
    features = ['HELLO_INTERVAL', 'TC_INTERVAL', 'WINDOW_SIZE', 'AVG_NEIGHBORS', 'STD_NEIGHBORS']
    distribution_info = {}

    for feature in features:
        print(f"\nAnalyzing feature: {feature}")
        feature_data = data[feature].dropna()

        # Compute Histogram Data
        counts, bin_edges = np.histogram(feature_data, bins=15)
        histogram_data = [{"bin": f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}", "count": int(counts[i])} for i in range(len(counts))]

        # Compute KDE Data
        kde = stats.gaussian_kde(feature_data)
        kde_x = np.linspace(feature_data.min(), feature_data.max(), 100)
        kde_y = kde(kde_x)
        kde_data = [{"x": round(float(x), 2), "y": round(float(y), 6)} for x, y in zip(kde_x, kde_y)]

        # Normality Test
        stat, p_value = stats.shapiro(feature_data)
        normality_result = 'Normal' if p_value > 0.05 else 'Non-Normal'

        # ✅ Improved Uniformity Test (KS-Test with correct parameters)
        uniform_params = stats.uniform.fit(feature_data)
        uniform_stat, uniform_p_value = stats.kstest(feature_data, 'uniform', args=uniform_params)
        uniformity_result = 'Uniform' if uniform_p_value > 0.05 else 'Non-Uniform'

        # ✅ Additional Chi-Square Test for Uniformity
        expected_freq = [np.mean(counts)] * len(counts)  # Expected uniform frequency
        chi_stat, chi_p_value = stats.chisquare(counts, expected_freq)
        chi_uniformity_result = 'Uniform' if chi_p_value > 0.05 else 'Non-Uniform'

        # Exponential Test (KS-Test with correct parameters)
        exp_params = stats.expon.fit(feature_data)
        exp_stat, exp_p_value = stats.kstest(feature_data, 'expon', args=exp_params)
        exp_result = 'Exponential' if exp_p_value > 0.05 else 'Non-Exponential'

        # Log-Normal Test
        shape, loc, scale = stats.lognorm.fit(feature_data, floc=0)
        lognorm_stat, lognorm_p_value = stats.kstest(feature_data, 'lognorm', args=(shape, loc, scale))
        lognorm_result = 'Log-normal' if lognorm_p_value > 0.05 else 'Non-Log-normal'

        # Gamma Test
        alpha, loc, beta = stats.gamma.fit(feature_data)
        gamma_stat, gamma_p_value = stats.kstest(feature_data, 'gamma', args=(alpha, loc, beta))
        gamma_result = 'Gamma' if gamma_p_value > 0.05 else 'Non-Gamma'

        # ✅ Store results in dictionary (FULL OUTPUT FORMAT)
        distribution_info[feature] = {
            "histogram": histogram_data,
            "kde": kde_data,
            "tests": {
                "normal": normality_result,
                "uniform": uniformity_result,
                "exponential": exp_result,
                "lognormal": lognorm_result,
                "gamma": gamma_result
            }
        }

    return distribution_info  # Full JSON output for frontend


# Step 2: Generate synthetic data with a different distribution (e.g., normal distribution)
def generate_subset_with_different_distribution(original_data: pd.DataFrame, features: list, size: int , distribution_type: str = 'uniform') -> pd.DataFrame:
    """
    Generates a subset of data with a different distribution for testing.
    
    :param original_data: The original dataset
    :param features: List of feature names to be modified
    :param size: The number of samples to generate in the new subset
    :param distribution_type: Type of distribution to use ('uniform', 'normal', etc.)
    :return: A modified dataframe with the new synthetic data
    """
    modified_data = original_data.copy()

    for feature in features:
        # Get the range of the feature (min, max)
        feature_min = original_data[feature].min()
        feature_max = original_data[feature].max()

        # Generate new values based on the chosen distribution type
        if distribution_type == 'uniform':
            modified_data[feature] = np.random.uniform(feature_min, feature_max, size)
        elif distribution_type == 'normal':
            mean = original_data[feature].mean()
            std_dev = original_data[feature].std()
            modified_data[feature] = np.random.normal(mean, std_dev, size)
        elif distribution_type == 'exponential':
            scale = original_data[feature].mean()  # Using mean as scale
            modified_data[feature] = np.random.exponential(scale, size)
        elif distribution_type == 'lognormal':
            shape, loc, scale = stats.lognorm.fit(original_data[feature].dropna(), floc=0)
            modified_data[feature] = np.random.lognormal(shape, loc, scale, size)
        elif distribution_type == 'gamma':
            alpha, loc, beta = stats.gamma.fit(original_data[feature].dropna())
            modified_data[feature] = np.random.gamma(alpha, beta, size)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    return modified_data



# Step 2: Based on the distribution analysis, suggest a new distribution and generate synthetic data
def generate_synthetic_data_from_suggestions(original_data: pd.DataFrame, distribution_info: dict, features: list, size: int = 1000):
    modified_data = original_data.copy()

    # Modify each feature based on the distribution analysis
    for feature in features:
        # Suggest new distributions based on the original distribution
        if distribution_info[feature]['normal'] == 'Normal':
            # Change normal distribution to uniform
            feature_min = original_data[feature].min()
            feature_max = original_data[feature].max()
            modified_data[feature] = np.random.uniform(feature_min, feature_max, size)
        elif distribution_info[feature]['uniform'] == 'Uniform':
            # Change uniform distribution to normal
            mean = original_data[feature].mean()
            std_dev = original_data[feature].std()
            modified_data[feature] = np.random.normal(mean, std_dev, size)
        elif distribution_info[feature]['exponential'] == 'Exponential':
            # Change exponential to log-normal
            shape, loc, scale = stats.lognorm.fit(original_data[feature].dropna(), floc=0)
            modified_data[feature] = np.random.lognormal(shape, loc, scale, size)
        elif distribution_info[feature]['lognormal'] == 'Log-normal':
            # Change log-normal to normal
            mean = original_data[feature].mean()
            std_dev = original_data[feature].std()
            modified_data[feature] = np.random.normal(mean, std_dev, size)
        elif distribution_info[feature]['gamma'] == 'Gamma':
            # Change gamma distribution to normal
            mean = original_data[feature].mean()
            std_dev = original_data[feature].std()
            modified_data[feature] = np.random.normal(mean, std_dev, size)
        else:
            # Keep the feature as is if no transformation is necessary
            pass

    return modified_data

# Example usage:
# Generate a synthetic dataset where the 'hello_interval' and 'tc_interval' features are normal instead of the original distribution



# Step 3: Model Evaluation

def evaluate_rf_model(original_data: pd.DataFrame, modified_data: pd.DataFrame, features: List[str], target: str):
    """
    Evaluates a pre-trained Random Forest regression model on both the original and modified datasets and compares results.
    
    :param original_data: The original dataset
    :param modified_data: The modified synthetic dataset
    :param features: List of feature names used for prediction
    :param target: The target variable
    :param model_path: Path to the pre-trained Random Forest model file
    :return: None
    """
    

    model_path = os.getenv('MODEL_PATH')
    if model_path is None:
        raise ValueError("MODEL_PATH environment variable not set.")
    print("loaded the model")
    model = load_rf_model(model_path)
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the path for the plots folder inside the current directory
    plots_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plots_dir):
      os.makedirs(plots_dir)  # Create the plots directory if it doesn't exist

   
    print("# Split original data into training and test sets")
    # Split original data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(original_data[features], original_data[target], test_size=0.3, random_state=42)
    
    print("# Split modified data into training and test sets")
    # Split modified data into training and test sets
    X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(modified_data[features], modified_data[target], test_size=0.3, random_state=42)
    
    print("# Evaluate the model on original test set")
    # Evaluate the model on original test set
    y_pred = model.predict(X_test)
    
    print("# Calculate RMSE and R² for the original data")
    # Calculate MSE and R² for the original data
    rmse_original = mean_squared_error(y_test, y_pred,squared=False)
    r2_original = r2_score(y_test, y_pred)
    print(f"Root Mean Squared Error (Original): {rmse_original:.4f}")
    print(f"R² Score (Original): {r2_original:.4f}")
    
    print("# Evaluate the model on modified synthetic test set")
    # Evaluate the model on modified synthetic test set
    y_pred_mod = model.predict(X_test_mod)

    # Calculate MSE and R² for the modified data
    rmse_modified = mean_squared_error(y_test_mod, y_pred_mod, squared=False)
    r2_modified = r2_score(y_test_mod, y_pred_mod)
    print(f"Root Mean Squared Error (Modified): {rmse_modified:.4f}")
    print(f"R² Score (Modified): {r2_modified:.4f}")
    print(f"Current directory: {current_dir}")
    print(f"Plots directory: {plots_dir}")

    mse_plot_path = os.path.join(plots_dir, 'rmse_comparison.png')
    # Visualize and save the comparison of MSE for original and modified data
    plt.figure()
    sns.barplot(x=['Original Data', 'Modified Data'], y=[rmse_original, rmse_modified])
    plt.title('RMSE Comparison: Original vs. Modified Data')
    plt.ylabel('Root Mean Squared Error')
    plt.savefig(mse_plot_path)  # Save the plot to a file
    plt.close()  # Close the plot to prevent it from displaying again

    # Visualize and save the comparison of R² scores for original and modified data
    # Save the R² comparison plot
    r2_plot_path = os.path.join(plots_dir, 'r2_comparison.png')
    plt.figure()
    sns.barplot(x=['Original Data', 'Modified Data'], y=[r2_original, r2_modified])
    plt.title('R² Comparison: Original vs. Modified Data')
    plt.ylabel('R² Score')
    plt.savefig(r2_plot_path)  # Save the plot to a file
    plt.close()  # Close the plot to prevent it from displaying again

    # Return the evaluation metrics and any additional data
    return {
        "rmse_original": rmse_original,
        "r2_original": r2_original,
        "rmse_modified": rmse_modified,
        "r2_modified": r2_modified,
        "mse_plot_path": mse_plot_path,  # Path to the saved MSE plot
        "r2_plot_path": r2_plot_path,    # Path to the saved R² plot
    }

    # Make retraining decision based on MSE comparison (lower is better for MSE)
    #should_retrain_rf(mse_original, mse_modified,modified_data,features,target)



def evaluate_linear_model(original_data: pd.DataFrame, modified_data: pd.DataFrame, features: List[str], target: str):
    """
    Evaluates the RCL calculation on both the original and modified datasets and compares results.

    :param original_data: The original dataset
    :param modified_data: The modified synthetic dataset
    :param features: List of feature names used for prediction
    :param target: The target variable
    :return: Dictionary containing evaluation metrics and plot paths
    """
    
    # Get the current directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the path for the plots folder inside the current directory
    plots_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)  # Create the plots directory if it doesn't exist

    def make_predictions(data: pd.DataFrame):
        predictions = []
        for _, row in data.iterrows():
            try:
                h = row['hello_interval']
                t = row['tc_interval']
                w = row['window_size']
                avg_neighbors = row['avg_neighbors']
                std_neighbors = row['std_neighbors']
                prediction = calculate_rcl(h, t, w, avg_neighbors, std_neighbors)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error calculating RCL for row: {row} - {str(e)}")
                predictions.append(None)
        return np.array(predictions)

    # Predict on original dataset
    y_pred_original = make_predictions(original_data[features])
    mse_original = mean_squared_error(original_data[target], y_pred_original)
    r2_original = r2_score(original_data[target], y_pred_original)
    
    # Predict on modified synthetic dataset
    y_pred_modified = make_predictions(modified_data[features])
    mse_modified = mean_squared_error(modified_data[target], y_pred_modified)
    r2_modified = r2_score(modified_data[target], y_pred_modified)
    
    # Generate and save MSE comparison plot
    mse_plot_path = os.path.join(plots_dir, 'rcl_mse_comparison.png')
    plt.figure()
    sns.barplot(x=['Original Data', 'Modified Data'], y=[mse_original, mse_modified])
    plt.title('RCL Calculation MSE Comparison: Original vs. Modified Data')
    plt.ylabel('Mean Squared Error')
    plt.savefig(mse_plot_path)
    plt.close()
    
    # Generate and save R² comparison plot
    r2_plot_path = os.path.join(plots_dir, 'rcl_r2_comparison.png')
    plt.figure()
    sns.barplot(x=['Original Data', 'Modified Data'], y=[r2_original, r2_modified])
    plt.title('RCL Calculation R² Comparison: Original vs. Modified Data')
    plt.ylabel('R² Score')
    plt.savefig(r2_plot_path)
    plt.close()
    
    return {
        "mse_original": mse_original,
        "r2_original": r2_original,
        "mse_modified": mse_modified,
        "r2_modified": r2_modified,
        "mse_plot_path": mse_plot_path,
        "r2_plot_path": r2_plot_path,
    }


    # Make retraining decision based on MSE comparison (lower is better for MSE)
   



def run_full_workflow_with_linear():
        # Step 1: Load the original dataset
        # Load the original dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'Total_File.xlsx')
        original_data = pd.read_excel(file_path, sheet_name='Sheet1')

        # Step 2: Analyze the original dataset's distributions
        print("Analyzing distribution of original data...")
        distribution_info = analyze_distribution(original_data)

        # Step 3: Generate synthetic data with a different distribution based on the analysis
        print("Generating synthetic data with different distributions based on analysis...")
        # modified_data = generate_synthetic_data_from_suggestions(original_data, distribution_info,features=['AVG_NEIGHBORS', 'STD_NEIGHBORS'], size=1000)
        modified_data = generate_subset_with_different_distribution(original_data,features=['AVG_NEIGHBORS', 'STD_NEIGHBORS'], size=4200,distribution_type= 'normal')
        print("Evaluating Random Forest model on original and synthetic datasets...")
        evaluate_rf_model(original_data, modified_data, features=['HELLO_INTERVAL', 'TC_INTERVAL', 'WINDOW_SIZE', 'AVG_NEIGHBORS', 'STD_NEIGHBORS'], target=["AVERAGE_OVERHEAD"])
     


        print("Workflow complete.")

def run_full_workflow_react():
    try:
        # Step 1: Load original dataset
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'Total_File.xlsx')
        original_data = pd.read_excel(file_path, sheet_name='Sheet1')

        # Step 2: Analyze original dataset's distributions
        original_distribution_info = analyze_distribution(original_data)

        # Step 3: Generate synthetic data
        modified_data = generate_subset_with_different_distribution(
            original_data, 
            features=['AVG_NEIGHBORS', 'STD_NEIGHBORS'], 
            size=4200, 
            distribution_type='uniform'
        )

        # Step 4: Analyze modified dataset's distributions
        modified_distribution_info = analyze_distribution(modified_data)

        # Step 5: Evaluate model
        results = evaluate_rf_model(
            original_data, 
            modified_data, 
            features=['HELLO_INTERVAL', 'TC_INTERVAL', 'WINDOW_SIZE', 'AVG_NEIGHBORS', 'STD_NEIGHBORS'], 
            target=["AVERAGE_OVERHEAD"]
        )

        return {
            "original_distribution_info": original_distribution_info,
            "modified_distribution_info": modified_distribution_info,
            "evaluation_results": results
        }

    except Exception as e:
        return {"error": str(e)}