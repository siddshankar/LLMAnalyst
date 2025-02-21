import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def analyze_housing_prices(csv_file, axioms):
    """
    Analyzes housing prices based on the given dataset and axioms using XGBoost and statistical analysis.

    Args:
        csv_file (str): Path to the CSV file containing the housing price data.
        axioms (list): A list of strings representing the axioms to test.

    Returns:
        dict: A dictionary containing the analysis results.
    """

    try:
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded successfully. Shape: {df.shape}, Columns: {df.columns.tolist()}")  # Initial information

        # --- Data Preprocessing ---

        # Rename columns to be lowercase and use underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]


        # Handle missing values (impute with mean/median/mode or drop)
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())  # Impute with median for numerical columns, more robust to outliers
                    print(f"Imputed missing values in column '{col}' with median.")
                else:
                    df[col] = df[col].fillna(df[col].mode()[0]) #Impute with mode for categorical columns
                    print(f"Imputed missing values in column '{col}' with mode.")

        # Encode categorical variables
        for col in df.select_dtypes(include=['object']).columns:
             try:
                df[col] = df[col].astype('category')
                df[col] = df[col].cat.codes
                print(f"Encoded categorical column: {col}")
             except:
                print(f"Could not encode column: {col}")

        # Calculate VIF before outlier handling to see initial multicollinearity
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        vif_data = pd.DataFrame()
        vif_data["feature"] = numerical_cols
        vif_data["VIF"] = [variance_inflation_factor(df[numerical_cols].values, i)
                                  for i in range(len(numerical_cols))]

        print("VIF before outlier handling:\n", vif_data)

        # Outlier detection and handling (using IQR method)
        for col in df.select_dtypes(include=np.number).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Clipping instead of removing. Removing can lead to data loss
            df[col] = df[col].clip(lower_bound, upper_bound)
            print(f"Handled outliers in column: {col}")
        
        #Feature Scaling
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # --- Axiom Evaluation ---

        axiom_results = {}
        for axiom in axioms:
            axiom_results[axiom] = "Analysis pending" # Placeholder, analysis for each axiom will be implemented below.
            # Example Axiom 1: 'Square footage is correlated with number of bedrooms and therefore house price'
            if axiom == 'Square footage is correlated with number of bedrooms and therefore house price':
                try:
                    correlation_sqft_bedrooms = df['sqft_living'].corr(df['bedrooms'])
                    correlation_sqft_price = df['sqft_living'].corr(df['price'])
                    correlation_bedrooms_price = df['bedrooms'].corr(df['price'])

                    # Consider interaction term: sqft_living * bedrooms
                    df['sqft_living_x_bedrooms'] = df['sqft_living'] * df['bedrooms']
                    correlation_interaction_price = df['sqft_living_x_bedrooms'].corr(df['price'])

                    # Check for non-linearity using polynomial features
                    df['sqft_living_squared'] = df['sqft_living']**2
                    correlation_sqft_squared_price = df['sqft_living_squared'].corr(df['price'])

                    # Visualization (optional)
                    #sns.regplot(x='sqft_living', y='bedrooms', data=df)
                    #plt.title('Sqft Living vs. Bedrooms')
                    #plt.show()

                    #sns.regplot(x='sqft_living', y='price', data=df)
                    #plt.title('Sqft Living vs. Price')
                    #plt.show()

                    #sns.regplot(x='bedrooms', y='price', data=df)
                    #plt.title('Bedrooms vs. Price')
                    #plt.show()

                    #sns.regplot(x='sqft_living_x_bedrooms', y='price', data=df)
                    #plt.title('Sqft Living * Bedrooms vs. Price')
                    #plt.show()


                    axiom_results[axiom] = {
                        'correlation_sqft_bedrooms': correlation_sqft_bedrooms,
                        'correlation_sqft_price': correlation_sqft_price,
                        'correlation_bedrooms_price': correlation_bedrooms_price,
                        'correlation_interaction_price': correlation_interaction_price,
                        'correlation_sqft_squared_price': correlation_sqft_squared_price
                    }

                    print(f"Correlation between sqft_living and Bedrooms: {correlation_sqft_bedrooms}")
                    print(f"Correlation between sqft_living and Price: {correlation_sqft_price}")
                    print(f"Correlation between Bedrooms and Price: {correlation_bedrooms_price}")
                    print(f"Correlation between (sqft_living * bedrooms) and Price: {correlation_interaction_price}")
                    print(f"Correlation between sqft_living^2 and Price: {correlation_sqft_squared_price}")

                except KeyError as e:
                    print(f"Column not found for axiom '{axiom}': {e}")
                    axiom_results[axiom] = "Error: Required columns not found"
                    
             # Example Axiom 2: 'Location is also important as it can increase or decrease crime rate and house size and therefore prices'
            if axiom == 'Location is also important as it can increase or decrease crime rate and house size and therefore prices':
                 try:
                    # Assuming zipcode represents location. Crime and House Size data is not there.
                    correlation_location_price = df['zipcode'].corr(df['price'])

                    # Group by zipcode and calculate mean price
                    location_price = df.groupby('zipcode')['price'].mean().sort_values(ascending=False)
                    print("Average price by zipcode: \n", location_price)


                    # Visualization (optional)
                    #sns.boxplot(x='zipcode', y='price', data=df)
                    #plt.title('Price Distribution by Zipcode')
                    #plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
                    #plt.tight_layout() # Adjust layout to prevent labels from overlapping
                    #plt.show()

                    # Investigate the relationship between latitude/longitude and price
                    correlation_lat_price = df['lat'].corr(df['price'])
                    correlation_long_price = df['long'].corr(df['price'])

                    #sns.scatterplot(x='lat', y='price', data=df)
                    #plt.title('Latitude vs. Price')
                    #plt.show()

                    #sns.scatterplot(x='long', y='price', data=df)
                    #plt.title('Longitude vs. Price')
                    #plt.show()


                    axiom_results[axiom] = {
                        'correlation_location_price': correlation_location_price,
                         'location_price_by_zipcode':location_price.to_dict(),
                        'correlation_lat_price': correlation_lat_price,
                        'correlation_long_price': correlation_long_price
                    }
                    print(f"Correlation between zipcode and Price: {correlation_location_price}")
                    print(f"Correlation between Latitude and Price: {correlation_lat_price}")
                    print(f"Correlation between Longitude and Price: {correlation_long_price}")
                 except KeyError as e:
                    print(f"Column not found for axiom '{axiom}': {e}")
                    axiom_results[axiom] = "Error: Required columns not found"

        # --- XGBoost Model ---
        X = df.drop('price', axis=1, errors='ignore')
        y = df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"RMSE: {rmse}")

        feature_importance = xgb_model.feature_importances_
        feature_names = X.columns
        feature_importance_dict = dict(zip(feature_names, feature_importance))
        print("Feature Importances:", feature_importance_dict)

        # Calculate VIF after outlier handling and scaling
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                                  for i in range(len(X.columns))]
        print("VIF after outlier handling and scaling:\n", vif_data)


        return {
            'axiom_results': axiom_results,
            'rmse': rmse,
            'feature_importance': feature_importance_dict
        }

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file}")
        return {'error': 'File not found'}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    csv_file = '/Users/sids/.cache/kagglehub/datasets/sukhmandeepsinghbrar/housing-price-dataset/versions/1/Housing.csv'  # Replace with your actual file path
    axioms = [
        'Square footage is correlated with number of bedrooms and therefore house price',
        'Location is also important as it can increase or decrease crime rate and house size and therefore prices'
    ]
    results = analyze_housing_prices(csv_file, axioms)
    print("Analysis Results:")
    print(results)