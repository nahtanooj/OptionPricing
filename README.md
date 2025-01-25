# Implementation and Simulation of Stochastic Models in Finance using Python

## Project Overview

This project focuses on the implementation and simulation of stochastic models used in quantitative finance, specifically the Black-Scholes, Heston, Dupire, and SABR models. The goal is to simulate asset price trajectories, calibrate these models using real market data, and compare their performance.

## Objectives

1. **Implementation of Stochastic Models**:
   - Implement and simulate the following stochastic models:
     - **Black-Scholes Model**: Based on geometric Brownian motion.
     - **Heston Model**: A stochastic volatility model.
     - **Dupire Model**: A local volatility model.
     - **SABR Model**: Used for modeling implied volatility in options markets.
   - Calibrate the models using real market data.
   - Compare the results of the simulations.

## Project Structure

The project is organized as follows:
```
PFE/
├── models/                     # Folder for model implementations
│   ├── black_scholes.py        # Black-Scholes model
│   ├── heston.py               # Heston model
│   ├── dupire.py               # Dupire model
│   ├── sabr.py                 # SABR model
│   └── __init__.py             # Make the folder a Python package
│
├── calibration/                # Folder for calibration logic
│   ├── calibration.py          # Calibration class
│   └── __init__.py             # Make the folder a Python package
│
├── visualization/              # Folder for visualization logic
│   ├── graphics.py             # Graphics class
│   └── __init__.py             # Make the folder a Python package
│
├── data/                       # Folder for market data
│   ├── market_data.csv         # Example market data file
│   └── __init__.py             # Make the folder a Python package
│
├── utils/                      # Folder for utility functions
│   ├── helpers.py              # Helper functions (e.g., for simulations, calculations)
│   └── __init__.py             # Make the folder a Python package
│
├── tests/                      # Folder for unit tests
│   ├── test_black_scholes.py   # Tests for Black-Scholes
│   ├── test_heston.py          # Tests for Heston
│   └── __init__.py             # Make the folder a Python package
│
├── app.py                      # Streamlit app for user interaction
├── main.py                     # Main script to run simulations/calibrations
└── requirements.txt            # List of dependencies
```

## Methodology

1. **Literature Review**:
   - Study the theoretical foundations of the stochastic models (Black-Scholes, Heston, Dupire, SABR).
   - Summarize the equations and the idea of each model in Memoire.pdf.

2. **Model Implementation**:
   - Implement each model in Python.
   - Calibrate the models using real market data.

3. **Performance Evaluation**:
   - Analyze the accuracy of the models.
   - Discuss the limitations of the models in real-world market scenarios.

## Expected Results

- Optimized implementation of financial models in Python.
- A final report analyzing the accuracy of the models and the performance of the algorithms in various market scenarios.

## Authors
Jonathan Level and Mohamed Naama