import pandas as pd
import numpy as np

salesData = pd.read_csv("sales.csv")
salesData['week_start'] = salesData['week_start'].astype(str)
salesData['week_start'] = pd.to_datetime(salesData['week_start'].str[:10], format='%Y-%m-%d')

salesData = salesData.groupby(['uniqueSKU', 'week_start']).agg({'totalBoxQty': 'sum'}).reset_index()

identifiers = pd.read_csv("unique_identifiers.csv")

salesData = salesData.merge(identifiers, on='uniqueSKU', how='left')
salesData.drop(columns=['uniqueSKU'], inplace=True)

details = pd.read_csv("SKUs_details.csv")
salesData = salesData.merge(details, on=['customerId', 'itemSkuNumber'], how='left')

column_names = ['week_start', 'qtySold', 'customerId', 'itemSkuNumber',
                'description', 'created_date', 'first_pick', 'type', 'skuLine']

salesData.columns = column_names


salesData = salesData.astype({
    'week_start': 'datetime64[ns]',
    'qtySold': 'int',
    'customerId': 'str',
    'itemSkuNumber': 'str',
    'description': 'str',
    'created_date': 'datetime64[ns]',
    'first_pick': 'datetime64[ns]',
    'type' : 'str',
    'skuLine': 'str'
})

salesData['month_created'] = salesData['created_date'].dt.strftime("%B")
salesData['sales_month'] = salesData['week_start'].dt.strftime("%B")
salesData['days_since_first_pick'] = ((salesData['week_start'] - salesData['first_pick']).dt.total_seconds() / (24 * 60 * 60)).round()
salesData.to_csv('data.csv', index=False)