import pandas as pd
import plotly.graph_objects as go

from utils.charts import create_bar_chart


def test_bar_chart_basic():
    # Minimal dataframe
    df = pd.DataFrame({
        "Department": ["HR", "Tech", "Marketing"],
        "Headcount": [12, 25, 8],
    })

    fig = create_bar_chart(df, x="Department", y="Headcount", title="Headcount by Department")

    # Basic assertions
    assert isinstance(fig, go.Figure)
    # Should have one trace (no grouping)
    assert len(fig.data) == 1
    # X labels should match departments
    x_vals = list(fig.data[0]["x"])
    assert x_vals == ["HR", "Tech", "Marketing"]
    # Y values should match headcounts
    y_vals = list(fig.data[0]["y"])
    assert y_vals == [12, 25, 8]
