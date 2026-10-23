<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=f5b042&height=120&section=header" alt="header"/> 
<h1 align="center">⚡Energy Consumption Time-Series Forecasting With Conformal Prediction</h1>
<h4 align="center">Status: <i>Conformal Prediction Ongoing</i></h4>
<h4 align="center">Expected Completion: <i>October 2026</i></h4>
<hr>

### NOTE: This README will summarize the whole pipeline. If you only want to see the final working product, jump to [Deployment](#deploy).

### Language Used: <a href="https://skillicons.dev"><img src="https://skillicons.dev/icons?i=py" /></a>

## 🏆 Key Results
* XGBoost model results in ~57.1Wh RMSE and 0.79 R<sup>2</sup> (explained variance) score.
* Adaptive conformal prediction covers 95% of the test data accurately with tight average margin of ~95.6Wh. The miscoverage mainly concentrates on sudden peak shifts.
* Implemented the model in an interactive dashboard in Streamlit Community Cloud (*Ongoing*). For now you can deploy locally only. I included guide to deploy [here](#deploy).
* The dashboard will simulate real-time forecasting by feeding the test set into the model row-by-row. You can adjust forecast horizon and playback speed using a slider.

## 📚 Table of Contents
* [Summary of Achievements](#summary-of-achievements)
* [Deployment](#deploy)
* [Future Update](#future-update)
* [Connect with the Author](#connect)

---

## About this Project
Predicting energy demand have been a crucial part of grid planning, which ensure that electricity supply are reliable while maintaining a minimum cost.

This project develops an end-to-end, real-time electricity consumption forecasting system using machine learning and conformal prediction. 

However, due to the limitation of not having live sensor and appliances reading, the final dashboard will only simulate real-time forecasting by providing the model with unseen data (testing set) row by row.


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
                The test result is:
                <table>
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R<sup>2</sup></th>
                    </tr>
                    <tr>
                        <td>XGBoost</td>
                        <td>57.13</td>
                        <td>28.23</td>
                        <td>0.79</td>
                    </tr>
                    <tr>
                        <td>ARIMAX</td>
                        <td>59.68</td>
                        <td>31.10</td>
                        <td>0.77</td>
                    </tr>
                </table>
                This results in XGBoost being chosen as the deployed model.
            </td>
        </tr>
        <tr>
            <td><a href = 'https://github.com/husnazainurin/energy-forecasting/blob/main/notebooks/04_conformal_prediction_analysis.ipynb'>Conformal Prediction</a></td>
            <td>
                MAPIE
            </td>
            <td>
                1. Applied and test adaptive conformal prediction on the selected model.<br><br>
                2. Achieved 95% coverage on test set, with miscoverage only concentrating around sudden peak shifts. In stable period, the interval remains tight and accurate.
            </td>
        </tr>
        <tr>
            <td><a href = https://github.com/husnazainurin/energy-forecasting/blob/main/src/app.py>Deployment</a></td>
            <td>streamlit</td>
            <td>
                Deployed the model in streamlit, which demonstrate the model forecasting in 2017 using sensor forecast from an external API. Using a slider, the user can adjust:<br><br>
                1. Forecast horizon: how far into the future they'd like to predict<br><br>
                2. Demo speed: how fast the demo iterate through the test set<br><br>
                And the dashboard will display:<br><br>
               1. Current time: current time, based on the current row being iterated in the test set<br><br>
               2. Current sensor reading: sensor reading fetch from live sensor<br><br>
               3. Predicted value: electrical consumption predicted at forecast horizon
            </td>
        </tr>
    </tbody>
</table>

## Project Structure
```plaintext
├── notebooks/          # Exploratory analysis and modeling
├── src/                # Streamlit app and helper scripts
├── models/             # My best trained model
├── data/processed      # Test dataset I included for your testing
├── requirements.txt    # Project dependencies
└── README.md
```

## Deployment <a id = 'deploy'></a>
If you want to view detailed codes, notebooks and techniques used, use [Local Deployment](#local).

Or, if you only have a few seconds to spare, I do plan on deploying the dashboard in Streamlit Community Cloud, but this will be done right after I'm satisfied with my current work. **Expected in late October/early November.**

### Local Deployment <a id = 'local'></a>

**Step 1**: Clone this repository
```bash
git clone https://github.com/husnazainurin/energy-forecasting.git
cd energy-forecasting
```

**Step 2**: Create and activate virtual environment

*Windows*
```bash
python -m venv .venv
.venv\Scripts\activate
```
*macOS/Linux*
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 3**: Install dependencies
```bash
pip install -r requirements.txt
```

**Step 4**: Launch Streamlit application
```bash
streamlit run src/app.py
```


## Future Update
This project have lots of improvement that is still within my capability, if I have extra time. My near-future plan is to:
* Include CNN in the model comparison, and switch the deployed model if it performs better than the XGBoost model

## Connect with the Author <a id = 'connect'></a>
If you like what I do, feel free to reach out to me via (*yes, the icon is clickable!!*) :
<div>
  <a href = 'https://www.linkedin.com/in/husna-zainurin-204558389'>
    <img width="48" height="48" src="https://github.com/user-attachments/assets/d3d48ae7-12ca-4e23-b746-c9fdc776540d"/>
  </a>
</div>


<img width=100% src="https://capsule-render.vercel.app/api?type=waving&color=f5b042&height=120&section=footer" alt="footer"/>
