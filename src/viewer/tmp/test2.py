import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit
from sklearn.linear_model import LinearRegression

# Sample data
df = pd.DataFrame(
    {
        "Category": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "X": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Y": [2, 4, 5, 4, 5, 6, 7, 8, 9],
    }
)

# Create a figure
fig = go.Figure()

# Iterate over each category
for category in df["Category"].unique():
    category_df = df[df["Category"] == category]

    # Create a scatter plot for the category
    fig.add_trace(
        go.Scatter(
            x=category_df["X"],
            y=category_df["Y"],
            mode="markers",
            name=category,
            marker=dict(size=10),
        )
    )

    # Fit a linear regression model
    X = category_df["X"].values.reshape(-1, 1)
    y = category_df["Y"].values
    model = LinearRegression().fit(X, y)

    # Create a line plot for the regression line
    x_line = np.linspace(min(X), max(X), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Fit for {category}",
            line=dict(dash="dash"),
        )
    )

# Customize the layout
fig.update_layout(
    title="Scatter Plot with Linear Fit Lines", xaxis_title="X", yaxis_title="Y"
)

fig.show()
