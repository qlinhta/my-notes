# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
from dash import Dash, html, dcc, dash_table
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas", "Apples"],
    "Amount": [4, 1, 2, 2, 4, 5, 2],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal", "NYC"]
})

app = Dash(__name__)

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig.update_layout(
    title="Fruit Eaten",
    xaxis_title="Fruit",
    yaxis_title="Number of Fruit Eaten",
    font=dict(
        family="SF Pro Display",
        size=18,
    )
)

scatter = px.scatter(df, x="Fruit", y="Amount", color="City")
scatter.update_layout(
    title="Fruit Eaten",
    xaxis_title="Fruit",
    yaxis_title="Number of Fruit Eaten",
    font=dict(
        family="SF Pro Display",
        size=18,
    )
)

# Create datatables from the DataFrame to display in the app
table = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)

app.head = [html.Link(rel='stylesheet', href='/assets/style.css')]

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),
    html.H3(children='''Dash: A web application framework for your data.'''),
    html.P(children='''Dash is a productive Python framework for building web applications.'''),
    dcc.Graph(
        id='01',
        figure=fig
    ),
    dcc.Graph(
        id='02',
        figure=scatter
    ),
    table
])

if __name__ == '__main__':
    app.run_server(debug=True)
