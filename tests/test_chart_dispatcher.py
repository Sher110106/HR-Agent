import pandas as pd
import plotly.graph_objects as go

from utils.charts import ChartSpec, create_chart


def test_dispatcher_bar_chart():
    df = pd.DataFrame({"Dept": ["HR", "Tech"], "Headcount": [10, 15]})
    spec = ChartSpec(chart_type="bar", x="Dept", y="Headcount", title="Headcount")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_dispatcher_line_chart():
    df = pd.DataFrame({"Month": ["Jan", "Feb"], "Sales": [100, 120]})
    spec = ChartSpec(chart_type="line", x="Month", y="Sales", title="Sales")
    fig = create_chart(df, spec)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


def test_dispatcher_raises_on_bad_type():
    df = pd.DataFrame({"A": [1], "B": [2]})
    spec = ChartSpec(chart_type="unknown", x="A", y="B")  # type: ignore[arg-type]
    try:
        create_chart(df, spec)
    except KeyError as e:
        assert "Unsupported" in str(e)
    else:
        raise AssertionError("Expected KeyError for unsupported chart type")
