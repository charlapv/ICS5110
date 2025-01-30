import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Load the trained models & transformers
linear_model = joblib.load("linear_model.pkl")
poly_model = joblib.load("poly_model.pkl")
poly_features = joblib.load("poly_features.pkl")
scaler = joblib.load("scaler.pkl")  # Load the saved StandardScaler
knn_model = joblib.load("KNN.pkl")
randforests_model = joblib.load("random_forests.pkl")

# Function to load and preview CSV data
def load_data(file):
    df = pd.read_csv(file)
    print("DEBUG: CSV Data in Gradio:\n", df.head())  # Print first 5 rows
    print("DEBUG: Data Types in Gradio:\n", df.dtypes)  # Check column types
    return df.head()  # Show first 5 rows

# Function to visualize population trends
def plot_population_trend(file, model_choice):
    df = pd.read_csv(file)

    plt.figure(figsize=(8,5))
    sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1])
    plt.xlabel("Years")
    plt.ylabel("Population")
    plt.title("Population Growth Trend")
    plt.grid()

    plt.figure(figsize=(8,5))
    plt.scatter(df["Year"], df["Population"], label="Actual Population", color="blue", alpha=0.6)

    X = df["Year"].values.reshape(-1, 1)  # Extract Year column

    if model_choice == "Linear Regression":
        X_scaled = scaler.transform(X) 
        predictions = linear_model.predict(X_scaled)
        plt.plot(df["Year"], predictions, label="Linear Regression", color="red", linestyle="dashed")
    elif model_choice == "Polynomial Regression": # Polynomial Regression
        X_scaled = scaler.transform(X)  # Apply scaling
        X_poly = poly_features.transform(X_scaled)  # Transform for Polynomial Regression
        predictions = poly_model.predict(X_poly)
        plt.plot(df["Year"], predictions, label="Polynomial Regression", color="green")
    elif model_choice == "KNN":  # K-Nearest Neighbors (KNN)
            predictions = knn_model.predict(X)
            label = "KNN"
            color = "blue"
            linestyle = "dotted"
            plt.plot(df["Year"], predictions, label="KNN", color="blue")
    else:   #Random Forests 
            predictions = randforests_model.predict(X)
            label = "Random Forests"
            color = "yellow"
            linestyle = "dotted"
            plt.plot(df["Year"], predictions, label="Random Forests", color="yellow")

    

    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.title(f"Population Growth Prediction ({model_choice})")
    plt.legend()
    plt.grid()

    #plt.savefig("population_trend.png")
    #return "population_trend.png"

    plt.savefig("population_trend.png")
    return "population_trend.png"

# Function to predict population using the selected model
def predict_population(file, model_choice):
    df = pd.read_csv(file)

    # Ensure correct column format
    if df.shape[1] < 2:
        return None, "ERROR: CSV must contain two columns (Year, Population).", None
    
    df.columns = ["Year", "Population"]
    df = df.astype({"Year": int, "Population": float})  # Convert data types

    X = df["Year"].values.reshape(-1, 1)  # Extract Year column

    if model_choice == "Linear Regression":
        # Do NOT scale X for Linear Regression
        X_scaled = scaler.transform(X) 
        predictions = linear_model.predict(X_scaled)  
    elif model_choice== "Polynomial Regression":  # Polynomial Regression
        X_scaled = scaler.transform(X)  # Apply the same scaling as training
        X_poly = poly_features.transform(X_scaled)  # Transform for Polynomial Regression
        predictions = poly_model.predict(X_poly)
    elif model_choice == "KNN":
         predictions = knn_model.predict(X)
    else:#random forests 
         predictions = randforests_model.predict(X)


    df["Predicted Population"] = predictions  # Append predictions to DataFrame

    # Extract the test set (2016-2020) before computing MSE & R²
    test_mask = df["Year"].between(2016, 2020)  # Select only test years
    X_test = df.loc[test_mask, "Year"].values.reshape(-1, 1)
    y_test = df.loc[test_mask, "Population"].values
    y_pred_test = df.loc[test_mask, "Predicted Population"].values

    # Compute metrics only on the test set
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print("DEBUG: Model Choice =", model_choice)
    print("DEBUG: X Values for Prediction:\n", X[:5])  # Print first 5 inputs
    print("DEBUG: Predictions:\n", predictions[:5])  # Print first 5 predictions

    return df, "population_trend.png", f"{model_choice} Results: MSE = {mse:.2f}, R² Score = {r2:.2f}"



# Wrapper function for Gradio
def gradio_interface(file, model_choice):
    preview = load_data(file)
    trend_image = plot_population_trend(file, model_choice)
    predictions, _, performance = predict_population(file, model_choice)
    return preview, trend_image, predictions, performance


# Define the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,  # Use the single wrapper function
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Radio(["Linear Regression", "Polynomial Regression","KNN", "Random Forests"], label="Choose Model")
    ],
    outputs=[
        gr.Dataframe(label="Preview Data"),
        gr.Image(label="Population Trend"),
        gr.Dataframe(label="Predictions"),
        gr.Textbox(label="Model Performance")
    ],
    title="Population Prediction Tool",
    description="Upload a CSV file with Year and Population data. Choose a model (Linear or Polynomial Regression, KNN or Random Forests) to predict future population trends."
)

# Launch the Gradio App
interface.launch()
