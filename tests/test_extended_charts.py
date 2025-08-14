import pandas as pd
import plotly.graph_objects as go

from utils.charts import ChartSpec, create_chart


def test_histogram_chart():
    df = pd.DataFrame({"Age": [25, 30, 35, 40, 45, 50]})
    spec = ChartSpec(chart_type="histogram", x="Age", title="Age Distribution")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_box_chart():
    df = pd.DataFrame({
        "Department": ["HR", "HR", "Tech", "Tech"],
        "Salary": [50000, 55000, 70000, 75000]
    })
    spec = ChartSpec(chart_type="box", x="Department", y="Salary", title="Salary by Dept")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # Box plot created


def test_violin_chart():
    df = pd.DataFrame({
        "Team": ["A", "A", "B", "B"],
        "Score": [85, 90, 75, 80]
    })
    spec = ChartSpec(chart_type="violin", x="Team", y="Score", title="Score Distribution")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # Violin plot created


def test_pie_chart():
    df = pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Count": [10, 20, 30]
    })
    spec = ChartSpec(chart_type="pie", x="Category", y="Count", title="Category Distribution")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1