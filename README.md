<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=f5b042&height=120&section=header" alt="header"/> 
<h1 align="center">⚡Energy Consumption Time-Series Forecasting With Conformal Prediction</h1>
<h4 align="center">Status: <i>Ongoing</i></h4>
<h4 align="center">Expected Completion: <i>October 2026</i></h4>
<hr>

### NOTE: This README will summarize the whole pipeline. If you only want to see the final working product, jump to [Deployment](#deploy).

### Language Used: <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=py" /></a>

## 📚 Table of Contents
* [Summary of Achievements](#summary-of-achievements)
* [Deployment](#deploy)
* [Future Update](#future-update)

---

## About this Project
Predicting energy demand have been a crucial part of grid planning, which ensure that electricity supply are reliable while maintaining a minimum cost. This project applies a time-series analysis to forecast electricity consumption using historical records and sensor reading. 

Dataset Used: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)

## Summary of Achievements
Here, I summarized the achievements I accomplished in several data science processes throughout the pipeline.

I only include the achievements here. For my thought process and justifications for these decisions, refer the <a href = 'notebooks'>notebooks</a> folder, or click on the data science process you'd like to view.
<table>
    <thead>
        <tr>
            <th>Data Science Process</th>
            <th>Libraries Used</th>
            <th>Summary</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/01_data_preprocessing.ipynb'>Data Wrangling</a></td>
            <td>
                pandas<br></td>
            <td>
              1. Convert 'date' column from str DataType to datetime64, and set it as index.<br><br>
              2. Extract calendar features (month, day, hour).<br><br>
              3. Drop columns representing readings from Chievres Airport.<br><br>
            </td>
        </tr>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/01_data_preprocessing.ipynb'>Data Preprocessing</a></td>
            <td>pandas</td>
            <td>
              1. No outlier handling done.<br><br>
              2. Aggregate highly correlated temperature and humidity features by location.<br><br>
            </td>
        </tr>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/02_eda.ipynb'>Exploratory Data Analysis (EDA)</a></td>
            <td>pandas, numpy, matplotlib, seaborn, statsmodel</td>
            <td>
                I've discovered these findings in the EDA process: <br><br>
                1. The data exhibit a strong daily seasonality, but weaker weekly seasonality. <br><br>
                2. Downsampling from 10-minutes intervals to 30-minutes interval reduce noise while preserving key signals. <br><br>
                3. Target feature (Appliances) at any given time is strongly influenced with the past version of itself, with influence persistence of 120-minutes.<br><br>
                4. Although influence decay past 120-minutes mark, the influence of past 1,5 and 7 days are relatively strong. This suggests seasonality that can be used for even more predictive power.<br><br>
                5. Lights, living room temperature and humidity, and outdoor temperature and humidity shows an immediate correlation to the target feature. Furthermore, lights have a short-term memory, while the others have a stable long-term influence.<br><br>
                6. Bathroom humidity exhibits a lagged correlation, starting past the 80-minutes mark, persisting until the 120-minutes mark.
            </td>
        </tr>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/02_eda.ipynb'>Feature Engineering</a></td>
            <td>pandas</td>
            <td>
                Based on EDA findings, I performed these feature engineering steps:<br><br>
                1. Keep hour as predictive feature, discard day.<br><br>
                2. Downsample to 30-minutes interval.<br><br>
                3. Create lagged variables for 10-120 minutes, as well as 1, 5, and 7 days.<br><br>
                4. Create short-term lagged feature for lights (10 - 60 minutes lag).<br><br>
                5. Create long-term lagged feature for bathroom humidity (80 - 120 minutes lag).<br><br>
                6. No lagged variables for living room and outdoor temperature and humidity.<br><br>
            </td>
        </tr>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/03_model_eda.ipynb'>Modelling</a></td>
            <td>
                scikit-learn, xgboost, optuna, pmdarima, statsmodel
            </td>
            <td>
                Using the feature engineered dataset, I modeled and tune:<br><br>
                1. XGBoost <br><br>
                2. ARIMAX with Fourier terms<br><br>
                <b>Results: </b> ARIMAX performs better with RMSE of ~94.5 Wh<br><br>
            </td>
        </tr>
    </tbody>
</table>



## Deployment <a id = 'deploy'></a>
If you just want to view it without any setup, use [Live Streamlit App Deployment](#live).

If you want to view detailed codes, notebooks and techniques used, use [Local Deployment](#local).

Or if you want neither, check out [this awesome demo video](#video) I made.

### Live Streamlit App Deployment <a id = 'live'></a>

### Local Deployment <a id = 'local'></a>

### Demo Video <a id = 'video'></a>

## Future Update
Currently, I only compared XGBoost Regressor and ARIMAX model. I do plan on including neural network model (specifically CNN) for the comparison and will replace the ARIMAX model if it perform better.


<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=f5b042&height=120&section=footer" alt="footer"/>
