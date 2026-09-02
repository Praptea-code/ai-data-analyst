"""Turn investigation chart specs (dicts) into renderable Plotly figures."""

import plotly.graph_objects as go


def spec_to_figure(spec: dict) -> go.Figure:
    """Build a Plotly Figure from a chart spec produced by the investigation graph.

    The spec carries real query results (x labels + y values). Chart type is a
    simple 'line' or 'bar' figure.
    """
    chart_type = spec.get("type", "bar")
    x = spec.get("x", [])
    y = spec.get("y", [])
    title = spec.get("title", "")
    y_label = spec.get("y_label", "value")
    x_label = spec.get("x_label", "item")

    fig = go.Figure()

    if chart_type == "line":
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=y_label))
    else:
        fig.add_trace(go.Bar(x=x, y=y, name=y_label))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
