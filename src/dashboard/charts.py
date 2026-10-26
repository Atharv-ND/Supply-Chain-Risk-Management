import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_late_pie_chart(df):
    status_counts = df['late_delivery_risk'].map({0: 'On Time', 1: 'Late'}).value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig = px.pie(status_counts, values='Count', names='Status',
                 title='Late vs On-Time Deliveries',
                 color='Status', color_discrete_map={'On Time': 'green', 'Late': 'red'}, hole=0.4)
    return fig

def plot_late_trend(df, time_agg='Monthly'):
    trend_df = df.copy()
    if time_agg == 'Monthly':
        trend_df['period'] = pd.to_datetime(trend_df['order_year'].astype(str) + '-' + trend_df['order_month'].astype(str) + '-01')
    else:
        trend_df['period'] = pd.to_datetime(trend_df['order_year'].astype(str) + '-01-01')
        
    trend_grp = trend_df.groupby('period').agg(
        total_shipments=('late_delivery_risk', 'count'),
        late_shipments=('late_delivery_risk', 'sum')
    ).reset_index()
    trend_grp['late_rate'] = trend_grp['late_shipments'] / trend_grp['total_shipments']
    
    fig = px.line(trend_grp, x='period', y='late_rate', title='Late Delivery Trend', markers=True)
    fig.update_yaxes(tickformat='.1%')
    return fig

def plot_bar_late_rate(df, category_col, title):
    grp = df.groupby(category_col).agg(
        total_shipments=('late_delivery_risk', 'count'),
        late_shipments=('late_delivery_risk', 'sum')
    ).reset_index()
    grp['late_rate'] = grp['late_shipments'] / grp['total_shipments']
    grp = grp.sort_values('late_rate', ascending=False)
    
    fig = px.bar(grp, x=category_col, y='late_rate', 
                 hover_data=['total_shipments'], title=title,
                 color='late_rate', color_continuous_scale='Reds')
    fig.update_yaxes(tickformat='.1%')
    return fig

def plot_scatter_lpi(df):
    grp = df.groupby('order_country').agg(
        total_shipments=('late_delivery_risk', 'count'),
        late_shipments=('late_delivery_risk', 'sum'),
        lpi_overall=('lpi_overall', 'mean')
    ).reset_index()
    grp['late_rate'] = grp['late_shipments'] / grp['total_shipments']
    
    # Filter tiny samples
    grp = grp[grp['total_shipments'] > 10]
    
    fig = px.scatter(grp, x='lpi_overall', y='late_rate', 
                     size='total_shipments', color='late_rate', hover_name='order_country',
                     title='LPI vs Late Delivery Rate (Country level)',
                     color_continuous_scale='Reds')
    fig.update_yaxes(tickformat='.1%')
    return fig

def plot_heatmap_shipping_market(df):
    pivot = df.pivot_table(index='shipping_mode', columns='market', values='late_delivery_risk', aggfunc='mean')
    fig = px.imshow(pivot, labels=dict(x="Market", y="Shipping Mode", color="Late Rate"), 
                    x=pivot.columns, y=pivot.index, title="Shipping Mode × Market Late Rate Heatmap",
                    color_continuous_scale='Reds', aspect="auto")
    return fig
