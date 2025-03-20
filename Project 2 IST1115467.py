# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:10:23 2025

@author: anna_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash
import os
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from sklearn.preprocessing import StandardScaler

# Load data
# Load data
df_2017_2018 = pd.read_csv("Merged_data.csv", parse_dates=["datetime"])
df_2019 = pd.read_csv("Merged_data2019.csv", parse_dates=["datetime"])
df_2019 = df_2019.drop(columns=['predicted_Power_kW'])
# Merge all years into a single dataset
df_2017_2019 = pd.concat([df_2017_2018, df_2019], ignore_index=True)


# Fill missing time features
df_2017_2019['month'] = df_2017_2019['datetime'].dt.month
df_2017_2019['hour'] = df_2017_2019['datetime'].dt.hour
df_2017_2019['day'] = df_2017_2019['datetime'].dt.day

def compute_heating_cooling_days_for_missing(df):
    # Filter the 2019 data where the columns are NaN
    df_2019_missing = df[(df['datetime'].dt.year == 2019) & (df['Heating_Day_difference'].isna() | df['Cooling_Day_difference'].isna())]
    
    # Calculate Heating and Cooling degree days only for rows where they are missing
    daily_avg_temp = df_2019_missing.groupby(df_2019_missing["datetime"].dt.date)["temp_C"].transform("mean")
    
    # Calculate Heating and Cooling degree days for the filtered 2019 data
    df.loc[df_2019_missing.index, "Heating_Day_difference"] = (16 - daily_avg_temp).clip(lower=0)
    df.loc[df_2019_missing.index, "Cooling_Day_difference"] = (daily_avg_temp - 21).clip(lower=0)
    
    return df

# Apply the function
df_2017_2019 = compute_heating_cooling_days_for_missing(df_2017_2019)
 # Filter the data to include only 2019
df_2019_filtered = df_2017_2019[df_2017_2019['datetime'].dt.year == 2019]

# Load models
models = {}
model_files = {
    "Linear Regression": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\linear_regression_model.pkl",
    "Random Forest": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\random_forest_model.pkl",
    "Gradient Boosting": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\boosting_model.pkl"
}

for model_name, file_path in model_files.items():
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            models[model_name] = pickle.load(f)
    else:
        print(f"Warning: {file_path} not found. {model_name} will not be available.")

print(models)  # Add this line to see if models are loaded correctly


# Define a dictionary to map original feature names to display labels
feature_label_map = {
    "Power_kW": "Power [kW]",
    "Day": "Day or Nighttime",
    "weekday": "Weekday",
    "Holiday": "Holiday",
    "Heating_Day": "Heating Day",
    "Cooling_Day": "Cooling Day",
    "rain_mm/h" :"Rain [mm/h]",
    "temp_C" :"Temperature [C]",
    "windSpeed_m/s" :"Wind speed [m/s]",
    "windGust_m/s" :"Wind gust [m/s]",
    "pres_mbar" :"Pressure [mbar]",
    "solarRad_W/m2" :"Solar radiation [W/m2]",
    "Heating_Day_difference" :"Heating Day differnece",
    "Cooling_Day_difference" :"Cooling Day differnece",
    "temp_C_lag1" :"Temperature lag1 [C]",
    "windSpeed_m/s_lag1" :"Wind speed lag1 [m/s]",
    "HR_lag1" :"HR lag1",
    "Power_kW_lag1": "Power lag1 [kW]",
    "sin_hour": "Sin hour",
    "cos_hour": "Cos hour",
    "month": "Month",
    "temp_HR_interaction": "Temperature, HR interaction",
    "day": "Day Number",
}

footer = html.Div([
    html.P([
        "Website by Anna Brenner| If you wanna know more about Técnico click ",
        html.A("here", href="https://tecnico.ulisboa.pt/", target="_blank", style={"color": "white", "textDecoration": "none"})
    ], style={"color": "white", "textAlign": "center", "padding": "10px", "margin": "0"})
], style={"backgroundColor": "#004080", "position": "relative", "bottom": "0", "width": "100%", "height": "40px"})


# Initialize Dash app
app = dash.Dash(__name__)
server=app.server
app.layout = html.Div([
    # Header with Logo & Title in One Line
    
    html.Div([
       # IST Técnico Logo (Aligned Left)
        html.Img(
            src="https://diaaberto.tecnico.ulisboa.pt/files/sites/178/ist_a_rgb_pos-1.png",
            style={"width": "100px", "height": "90px", "margin-right": "20px", "background": "transparent"}
        ),

        # Title (Centered in Full-Width Blue Bar)
        html.H1("IST Central Building Energy Consumption", style={
            "color": "white",
            "margin": "0",
            "textAlign": "center",
            "flex": "1"
        })
    ], style={
        "backgroundColor": "#004080",  # Updated to match the logo color
        "display": "flex",
        "alignItems": "center",
        "padding": "15px",
        "width": "100%",
        "height": "60px"
    }),

    # Two Images Below (Location & Building)
    html.Div([
        html.Div([
            html.P("IST Campus Map", style={"textAlign": "center", "fontWeight": "bold"}),
            html.Img(
                src="https://www.ipfn.tecnico.ulisboa.pt/eftc2015/EFTC2015_webpage/Welcome_files/mapa-site-ist-EN.png",
                style={"width": "400px", "height": "350px"}
            ),
        ], style={"textAlign": "center", "margin": "10px"}),
        html.Div([
            html.P("IST Central Building", style={"textAlign": "center", "fontWeight": "bold"}),
            html.Img(
                src="https://tecnico.ulisboa.pt/files/2018/03/template_fotos-18-1140x641.jpg",
                style={"width": "400px", "height": "350px"}
            ),
        ], style={"textAlign": "center", "margin": "10px"})
    ], style={"display": "flex", "justifyContent": "center", "padding": "10px"}),
    

    # Create a row with feature selection and date range picker
    html.Div([
        # Feature selection dropdown
        html.Div([
            html.Label("Select Features"),
            dcc.Dropdown(
                id="feature-select",
                options=[{"label": feature_label_map.get(col, col), "value": col, "value": col} for col in df_2017_2019.columns if col != "datetime"],
                multi=True,
                value=["Power_kW"],
                style={"width": "100%", "hight":"60px"}
            ),
        ], style={"width": "50%", "margin-right": "10px", 'height': '100%','flexDirection': 'column'}),  # Adjusts spacing

        # Date range picker with improved navigation
        html.Div([
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id="date-range",
                start_date=df_2017_2019["datetime"].min().date(),
                end_date=df_2017_2019["datetime"].max().date(),
                display_format="YYYY-MM-DD",
                with_portal=True,  # Opens in full screen for easier selection
                start_date_placeholder_text="Start Date",
                end_date_placeholder_text="End Date"
            ),
        ], style={"width": "25%"}),  
        html.Div([html.Label("Select Graph Type:"),
        dcc.RadioItems(
            id='graph-type',
            options=[
                {'label': 'Line Chart', 'value': 'line'},
                {'label': 'Bar Chart', 'value': 'bar'},
                {'label': 'Scatter Plot', 'value': 'scatter'}
            ],
            value='line',
            inline=True
        )
            
            ],style={'width': '25%', 'padding': '10px', 'height': '100%'}),  # 32% width for Graph Type Select
    ], style={"display": "flex", "align-items": "center","justify-content": "space-between"}),  # Aligns elements in a row
   
    
    # Box for displaying the highest and lowest values
    html.Div(id="feature-summary", style={'display': 'flex', 'flexDirection': 'column', "align-items": "center",'marginBottom': '20px'}),

    
    
    dcc.Graph(
            id="time-series-graph",
            style={
                'width': '75%',  # Adjust the width of the graph
                'margin': '0 auto'  # Centers the graph horizontally
            }
        ),
    
    html.Div([
    html.Label("Select Regression Model:"),
    dcc.Dropdown(
        id="model-select",
        options=[{"label": model, "value": model} for model in models.keys()],
        value=list(models.keys())[0] if models else None,
        style={"width": "50%", "margin-right": "20px"}  # Adjust width and spacing
    ),
    
    html.Label("Select Metrics to Display:"),
    dcc.Checklist(
        id="metrics-select",
        options=[
            {"label": "Mean Absolute Error (MAE)", "value": "MAE"},
            {"label": "Mean Squared Error (MSE)", "value": "MSE"},
            {"label": "R² Score", "value": "R2"}
        ],
        value=["MAE", "MSE", "R2"],  # Default to showing all metrics
        style={"display": "flex", "flexDirection": "column"},  # Makes options vertical
        inline=True
    )
    
], style={"backgroundColor": "#d9d9d9", "borderTopLeftRadius": "15px", "borderTopRightRadius": "15px", "borderBottomLeftRadius": "0px", "borderBottomRightRadius": "0px", "padding": "20px","display": "flex", "align-items": "center", "gap": "10px"}),
    
   html.Div([
    # Graph on the left
    html.Div([
        dcc.Graph(id="prediction-graph", style={
            'width': '100%',  # Full width within its container
            'height': '100%'  # Ensures it stretches properly
        })
    ], style={'width': '75%', 'height': '100%', 'display': 'flex', 'margin-right': '20px','align-items': 'center'}),  # Left-side graph

    # Metrics on the right
    html.Div([
        html.H3("Model Performance Metrics", style={'font-size': '20px', 'font-weight': 'bold', 'text-align': 'left', 'width': '100%'}),  
        html.Div(id="metrics-table", style={'width': '100%',  "padding": "10px","textAlign": "center"})  # Metrics container
    ], style={'width': '25%','gap': "20px", 'height': '100%', 'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', "justify-content": "space-between"})  # Right-side metrics
], style={"backgroundColor": "#d9d9d9", "borderTopLeftRadius": "0px", "borderTopRightRadius": "0px", "borderBottomLeftRadius": "15px", "borderBottomRightRadius": "15px", "padding": "20px","display": "flex", "align-items": "flex_start", "justifyContent": "space-between", "height": "100%", "gap":"40px"}),  # Main flex container
 
footer
])

        
@app.callback(
    Output("time-series-graph", "figure"),
    [Input("feature-select", "value"), 
     Input("date-range", "start_date"), 
     Input("date-range", "end_date"),
     Input("graph-type", "value")]
)
def update_graph(selected_features, start_date, end_date, graph_type):
    fig = go.Figure()
    
    # Convert start_date and end_date to datetime
    if start_date and end_date:
        filtered_df = df_2017_2019[
            (df_2017_2019["datetime"] >= pd.to_datetime(start_date)) & 
            (df_2017_2019["datetime"] <= pd.to_datetime(end_date))
        ]
    else:
        filtered_df = df_2017_2019  # Use full dataset if no range is selected
    
    # If no features are selected, show a message
    if not selected_features:
        fig.add_annotation(
            text="No feature selected. Please select at least one feature to display the graph.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title="No Data to Display")
        return fig
    
    # Add one trace for each selected feature
    colors = ["darkblue", "red", "green", "purple", "orange", "violet",]  # Color options for different features
    feature_labels = []  # To store the mapped feature labels
    for i, feature in enumerate(selected_features):
        feature_label = feature_label_map.get(feature, feature)
        feature_labels.append(feature_label)
        y_axis = f"y{i+1}"
        if graph_type == "bar":
            fig.add_trace(go.Bar(
                x=filtered_df["datetime"], 
                y=filtered_df[feature], 
                name=feature_label,
                marker=dict(color=colors[i % len(colors)]), 
                yaxis=y_axis
            ))
        elif graph_type == "scatter":
            fig.add_trace(go.Scatter(
                x=filtered_df["datetime"], 
                y=filtered_df[feature], 
                mode="markers",
                name=feature_label,
                marker=dict(color=colors[i % len(colors)]),
                yaxis=y_axis
            ))
        else:  # Default to line graph
            fig.add_trace(go.Scatter(
                x=filtered_df["datetime"], 
                y=filtered_df[feature], 
                mode="lines", 
                name=feature_label,
                line=dict(color=colors[i % len(colors)]),
                yaxis=y_axis
            ))
    
    # Add layout options for multiple Y-axes
    layout = {
        'title': "Selected Features Over Time",
        'xaxis': {'title': 'Date'},
        'yaxis': {'title': feature_labels[0]},  # Default to the first feature for the first Y-axis
        'yaxis2': {}, 'yaxis3': {}, 'yaxis4': {},  # Additional y-axes for more than 2 features
        'showlegend': True,
        'barmode': 'stack',  # Ensure bars are stacked if multiple traces are added
        'margin': {'l': 60, 'r': 60, 't': 60, 'b': 60} 
    }
    
    # Dynamically configure multiple Y-axes based on the number of selected features
    for i in range(1, len(selected_features)):
        layout[f'yaxis{i+1}'] = {
            'title': feature_labels[i],
            'overlaying': 'y',  # Overlay on the first Y-axis
            'side': 'right' if i % 2 != 0 else 'left',  # Alternate between right and left sides for each axis
        }

    fig.update_layout(layout)
    return fig

@app.callback(
    Output("feature-summary", "children"),  # Update the feature summary box
    [Input("feature-select", "value"), Input("date-range", "start_date"), Input("date-range", "end_date")]
)
def update_feature_summary(selected_features, start_date, end_date):
    # Convert start_date and end_date to datetime
    if start_date and end_date:
        filtered_df = df_2017_2019[
            (df_2017_2019["datetime"] >= pd.to_datetime(start_date)) & 
            (df_2017_2019["datetime"] <= pd.to_datetime(end_date))
        ]
    else:
        filtered_df = df_2017_2019  # Use full dataset if no range is selected
    
    summary_boxes = []
    excluded_features = {"weekday", "Holiday", "hour", "Heating_Day", "Cooling_Day", "day", "month", "Day"}
    # Iterate over selected features and calculate the highest and lowest values
    for feature in selected_features:
        if feature in filtered_df.columns:
            # Map feature to the display name
            feature_label = feature_label_map.get(feature, feature)

            # Find highest and lowest values and the corresponding days
            max_value = filtered_df[feature].max()
            min_value = filtered_df[feature].min()
            max_day = filtered_df[filtered_df[feature] == max_value]['datetime'].iloc[0].date()
            min_day = filtered_df[filtered_df[feature] == min_value]['datetime'].iloc[0].date()

            # Create summary boxes for max and min values
            feature_summary = [
                html.Div(f"Highest {feature_label}: {max_value:.2f} on {max_day}", style={
                    'backgroundColor': 'Lightblue', 'padding': '10px', 'margin': '5px', 'flex': '1','border-radius': '15px',  # Round corners
                        'box-shadow': '2px 2px 6px rgba(0, 0, 0, 0.1)',  # Soft shadow
                        'max-width': '400px',  # Limit width of each box
                        'text-align': 'center',  # Center text inside the box
                        
                }),
                html.Div(f"Lowest {feature_label}: {min_value:.2f} on {min_day}", style={
                    'backgroundColor': 'lightgrey', 'padding': '10px', 'margin': '5px', 'flex': '1','border-radius': '15px',  # Round corners
                        'box-shadow': '2px 2px 6px rgba(0, 0, 0, 0.1)',  # Soft shadow
                        'max-width': '400px',  # Limit width of each box
                        'text-align': 'center',  # Center text inside the box
                        
                })
            ]
            
            # Add average value if the feature is not in the excluded features list
            if feature not in excluded_features:
                avg_value = filtered_df[feature].mean()
                feature_summary.append(
                    html.Div(f"Average {feature_label}: {avg_value:.2f}", style={
                        'backgroundColor': 'lightblue', 'padding': '10px', 'margin': '5px', 'flex': '1', 'border-radius': '15px',  # Round corners
                        'box-shadow': '2px 2px 6px rgba(0, 0, 0, 0.1)',  # Soft shadow
                        'max-width': '400px',  # Limit width of each box
                        'text-align': 'center',  # Center text inside the box
                       
                    })
                )
            
            # Add the feature summary to the list of summary boxes
            summary_boxes.append(
                html.Div(feature_summary, style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center','gap':'10px'})
            )
    
    return summary_boxes


@app.callback(
    [Output("prediction-graph", "figure"), Output("metrics-table", "children")],
    [Input("model-select", "value"),Input("metrics-select", "value")]
)
def update_predictions(selected_model, selected_metrics):
    # Check if selected model exists in the models dictionary
    if selected_model not in models:
        return go.Figure(), html.P("Selected model is not available.")
    
    # Define the model file paths
    model_file_paths = {
        "Linear Regression": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\linear_regression_model.pkl",
        "Random Forest": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\random_forest_model.pkl",
        "Gradient Boosting": r"C:\Users\anna_\Downloads\Project 1 Energy Services IST1115467\boosting_model.pkl"
    }
    
    # Load the appropriate model using the selected model's path
    model_path = model_file_paths[selected_model]
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # List of features used during training
    important_features = [ 'Power_kW_lag1','hour', 'cos_hour', 'solarRad_W/m2', 'temp_C' ]

    # Filter 2019 data and prepare for prediction
    df_2019_data = df_2017_2019[df_2017_2019['datetime'].dt.year == 2019]
    X_test = df_2019_data[important_features]  # Use the important features defined earlier
    y_actual = df_2019_data['Power_kW']  # Actual values for comparison
    
    # Recreate and fit the scaler (StandardScaler)
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)  # Fit and transform to the test data

    # Get predictions from the model
    y_pred = model.predict(X_test_scaled)

    # Create the figure to show actual vs predicted values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_2019_data['datetime'], y=y_actual,
                             mode='lines', name='Actual Consumption', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_2019_data['datetime'], y=y_pred,
                             mode='lines', name=f'Predicted ({selected_model})', line=dict(color='red')))
    
    fig.update_layout(title={
        'text': f"Predicted vs Actual Power Consumption - {selected_model}",
        'x': 0.5,  # Centers the title
        'xanchor': 'center'  # Ensures proper alignment
        },
                      xaxis_title="Date", yaxis_title="Power Consumption (kW)", legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.25,  # Moves the legend below the x-axis
            xanchor="center",
            x=0.5),
            )
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    # Create a dynamic metrics table based on user selection
    metrics_rows = [
        html.Tr([html.Th("Metric"), html.Th("Value")])  # Table header
    ]

    if "MAE" in selected_metrics:
        metrics_rows.append(
            html.Tr([
                html.Td(html.Span("Mean Absolute Error (MAE)",
                                  title="MAE measures the average absolute difference between actual and predicted values. Lower is better.")),
                html.Td(f"{mae:.2f}")
            ])
        )

    if "MSE" in selected_metrics:
        metrics_rows.append(
            html.Tr([
                html.Td(html.Span("Mean Squared Error (MSE)",
                                  title="MSE squares the errors before averaging, making larger errors more significant. Lower is better.")),
                html.Td(f"{mse:.2f}")
            ])
        )

    if "R2" in selected_metrics:
        metrics_rows.append(
            html.Tr([
                html.Td(html.Span("R² Score",
                                  title="R² (coefficient of determination) indicates how well the model explains the variance in the data. Closer to 1 is better.")),
                html.Td(f"{r2:.2f}")
            ])
        )

    metrics_table = html.Table(metrics_rows)

    return fig, metrics_table

# Layout using Flexbox to make the graph and table appear side by side
    layout = html.Div([
        # Graph container
        html.Div(dcc.Graph(figure=fig), style={
            'flex': '3',
            'width': '70%',
            'padding-right': '20px',  # Space between the graph and metrics table
        }),
        
        # Metrics table container
        html.Div(metrics_table, style={
            'flex': '1',
            'width': '30%',
            'padding-left': '20px',
            'padding-top': '40px',
            'display': 'flex',
            'justify-content': 'center',
        })
    ], style={
        'display': 'flex',
        'justify-content': 'space-between',  # Distribute space between graph and table
        'align-items': 'flex-start',  # Align both graph and table at the top
        'width': '100%'  # Ensure the container takes up full width
    })
    
    return layout
# Run the app
import webbrowser
webbrowser.open("http://127.0.0.1:8050/")

if __name__ == "__main__":
    app.run_server(debug=True)
