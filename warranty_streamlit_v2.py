import streamlit as st
import openpyxl
import pandas as pd
import numpy as np
import plotly_express as px
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
#from fbprophet import *
import fbprophet as Prophet
import itertools
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
import pystan
import holidays
import os
import sys
from glob import glob
from sklearn.metrics import mean_absolute_error as MAE
from scipy.stats import skew
from sklearn.metrics import r2_score
import altair as alt
import seaborn as sns
import datetime


today = datetime.date.today()

@st.cache(suppress_st_warning=True)
def read_data():
    sales_sheet = 'sales'
    claims_sheet = 'claims' 
    file_name = './../data/Sales_Warranty_Claims_clean.xlsx'
    claims_df = pd.read_excel(file_name, sheet_name = claims_sheet, header = 0, engine='openpyxl')
    sales_df = pd.read_excel(file_name, sheet_name = sales_sheet, header = 0, engine='openpyxl')
    return claims_df, sales_df

claims_df, sales_df = read_data()


cleansed_claims_df = claims_df.copy()
cleansed_sales_df = sales_df.copy()

cleansed_claims_df.drop(cleansed_claims_df.columns[[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]], axis=1, inplace=True)

cleansed_claims_df['Material'] = cleansed_claims_df['Mat_#_M+H_installed']

cleansed_claims_df['Mon_Year'] = cleansed_claims_df['Date_Veh_Repair_Y/M/D'].dt.strftime('%b %Y')

initial_merge_df = cleansed_sales_df.merge(cleansed_claims_df, how = 'outer', on = ['Material'])

print('initial_merge', initial_merge_df)

#### The interactive bit

st.title('Warranty forecasting')
'''

This webapp allows you to select, visualize and forecast warranty claims data.
'''

st.sidebar.title('Warranty claims historical and forecasting data')

st.sidebar.markdown('Interact with the data here')

start_date = st.sidebar.date_input('Start date', today)

start_date = pd.to_datetime(start_date)

initial_merge_202005 = initial_merge_df[initial_merge_df['Date_Veh_Repair_Y/M/D'] > start_date]

initial_merge_202005['freq']=initial_merge_202005.groupby(by='Material')['Material'].transform('count')

min_claims = st.sidebar.slider('How many claims in the Start date?', 10, 100, 5)

initial_merge_relevant = initial_merge_202005[initial_merge_202005['freq'] > min_claims]

material_list = initial_merge_relevant.Material.unique()

#st.sidebar.dataframe(initial_merge_relevant)

# Create a list of possible values and multiselect menu with them in it.
MATERIAL_SELECTED = st.sidebar.multiselect('Select Material', material_list)

#Sales_part_numbers = st.sidebar.multiselect('Select relevant Sales part number/s', )

warranty_data_plotting = st.sidebar.selectbox(
    'Historical data, Backtest or Forecast?',
     ['Historical_data', 'Backtest', 'Forecast'])

'You have selected: ', warranty_data_plotting

# Mask to filter dataframe to select only materials that have a minimum required number of claims made in the selected period
mask_materials = initial_merge_relevant['Material'].isin(MATERIAL_SELECTED)

initial_merge_relevant = initial_merge_relevant[mask_materials]

initial_merge_relevant['Veh_Repair_Date'] = pd.to_datetime(initial_merge_relevant['Date_Veh_Repair_Y/M/D'])

initial_merge_relevant = initial_merge_relevant.set_index(['Veh_Repair_Date'])

initial_merge_relevant.index = pd.to_datetime(initial_merge_relevant.index, unit='D')

print(initial_merge_relevant)

# Get monthly count of claims made for a given material, and classify based on repair date, country, months in service etc.
claims_frequency_df = initial_merge_relevant.groupby(['Mon_Year']).agg({'Material':'count'}).reset_index()

claims_frequency_df_country = initial_merge_relevant.groupby(['Veh_Repair_1_Country']).agg({'Material':'count'}).reset_index()

claims_frequency_df_country = claims_frequency_df_country.sort_values('Material', ascending=False)

claims_frequency_df_MIS = initial_merge_relevant.groupby(['Months_in_Service']).agg({'Material':'count'}).reset_index()

claims_frequency_df_MIS = claims_frequency_df_MIS.sort_values('Months_in_Service')

claims_frequency_df_MIS = claims_frequency_df_MIS[1:].reset_index()

claims_frequency_df_MIS = claims_frequency_df_MIS.astype({'Months_in_Service': 'int32'})

print('claims_frequency_df_MIS', claims_frequency_df_MIS)

# Prepare data for forecasting

cleansed_claims_df = cleansed_claims_df.dropna()

cleansed_claims_df = cleansed_claims_df.drop(['Claim_#', 'Veh_VIN_#', 'Mat_#_M+H_installed', 'Mat_#_Cust_installed', 'Date_Veh_Regist_Y/M/D', 'Alert_#', 'M+H_Plant_2_Comp_Code'], axis=1)

cleansed_claims_df_filtered = cleansed_claims_df[cleansed_claims_df['Material'].str.contains(MATERIAL_SELECTED[0])]

claims_frequency_df_fcst = cleansed_claims_df_filtered.groupby('Mon_Year')['Mon_Year'].agg(['count']).reset_index()

claims_frequency_df_fcst['Mon_Year'] = claims_frequency_df_fcst['Mon_Year'].str.upper()

claims_frequency_df_fcst.columns = ['date', 'claims_count']

#filtered sales df

cleansed_sales_df_filtered = cleansed_sales_df[cleansed_sales_df['Material'].str.contains(MATERIAL_SELECTED[0])]

cleansed_sales_df_filtered = cleansed_sales_df_filtered.dropna(axis=1, how='all')

# cleansed_sales_df_filtered = pd.melt(cleansed_sales_df_filtered, id_vars=['Material','Product Name', 'M+H_Proj_#', 'M+H_Plant_2_Comp_Code','Date_Sales_Y/M'], var_name='date')

# print('cleansed_sales_df_filtered', cleansed_sales_df_filtered)


if warranty_data_plotting == 'Historical_data':
    
    parameter = st.sidebar.selectbox(
        'Select parameter to plot',
        ['Repair Date', 'Country of Failure', 'Months in Service'])


    if parameter == 'Country of Failure':

        fig, ax = plt.subplots(figsize=(20,10))

        #sns.countplot(x='Mon_Year', data = initial_merge_relevant, ax = ax1)

        sns.barplot(y = 'Material', x='Veh_Repair_1_Country', data = claims_frequency_df_country)#, order=sorted(claims_frequency_df_country.Veh_Repair_1_Country))

        plt.xticks(rotation=90, fontsize=10)

        plt.xlabel('Country of repair', fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        st.pyplot(fig)

    elif parameter == 'Repair Date':

        fig, ax = plt.subplots(figsize=(20,10))

        sns.countplot(x='Mon_Year', data = initial_merge_relevant)

        plt.xticks(rotation=90, fontsize=10)

        plt.xlabel('Repair date', fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        st.pyplot(fig)

    elif parameter == 'Months in Service':

        fig, ax = plt.subplots(figsize=(20,10))

        sns.barplot(y = 'Material', x='Months_in_Service', data = claims_frequency_df_MIS)

        plt.xticks(rotation=90, fontsize=10)

        plt.xlabel('Months in service', fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        st.pyplot(fig)


#### Forecasting work

## ***********************

elif warranty_data_plotting == 'Backtest':

    include_sales_history = st.sidebar.checkbox("Include Sales history data as a factor?")

    print(include_sales_history)

    if include_sales_history == True: #len(cleansed_sales_df_filtered) > 5 & 

        sales_part_num = []

        sales_part_num.append(st.sidebar.text_input("Enter the sales part number/s (if more than one, comma-separated) you would like included")) # MATERIAL_SELECTED[0]

        sales_part_num = sales_part_num[0].split(",")
        
        print('Sales part numbers', sales_part_num)


        if (len(sales_part_num) == 1) and (sales_part_num[0] == MATERIAL_SELECTED[0]):
        
            cleansed_sales_df_filtered = cleansed_sales_df[cleansed_sales_df['Material'].str.contains(sales_part_num[0])]

            cleansed_sales_df_filtered = cleansed_sales_df_filtered.dropna(axis=1, how='all')

            cleansed_sales_df_filtered = pd.melt(cleansed_sales_df_filtered, id_vars=['Material','Product Name', 'M+H_Proj_#', 'M+H_Plant_2_Comp_Code','Date_Sales_Y/M'], var_name='date')
        
            merged_df = cleansed_sales_df_filtered.merge(claims_frequency_df_fcst, how = 'inner', on = ['date'])

            merged_df = merged_df.rename(columns={"value":"sales_volume"})

            #scaler = MinMaxScaler()

            #merged_df[['sales_volume', 'claims_count']] = scaler.fit_transform(merged_df[['sales_volume', 'claims_count']])

            merged_df = merged_df.drop(['Material', 'Product Name', 'M+H_Proj_#','Date_Sales_Y/M', 'M+H_Plant_2_Comp_Code'], axis=1)

            merged_df[['date']] = pd.to_datetime(merged_df['date'].astype(str))

            correlation = merged_df['sales_volume'].corr(merged_df['claims_count'])

            st.write('The correlation between sales and claims data is', correlation)

            print(merged_df)

            df_for_forecast = merged_df.copy()

            #Generate a sales forecast first
            df_sales = pd.DataFrame()
            df_sales['ds'] = df_for_forecast['date']
            df_sales['y'] = df_for_forecast['sales_volume']

            df_sales = df_sales[(df_sales['ds'] > '2016-01-01') & (df_sales['ds'] < '2021-05-01')].reset_index(drop=True)

            print('df_sales', df_sales)
            
            m_sales = Prophet.Prophet(changepoint_prior_scale = 0.95)

            m_sales.fit(df_sales)

            future_sales = m_sales.make_future_dataframe(periods=12,freq='MS')

            print('future dataframe', future_sales)

            sales_forecast = m_sales.predict(future_sales)

            print('future sales', sales_forecast)

            df = pd.DataFrame()
            df['ds'] = df_for_forecast['date']
            df['y'] = df_for_forecast['claims_count']
            df['sales_volume'] = df_sales['y']

            df = df.sort_values('ds').reset_index(drop = True)

            df = df[df['ds'] < '2021-05-01']

            len_df = len(df)

            print(len_df)

            print("Trimmed df for 1354072S01", df)

            train_len = int(0.6*len_df)

            future_len = len_df - train_len

            print(train_len, future_len)

            train = df.head(train_len).dropna()

            train = train.sort_values('ds').reset_index(drop = True)

            print(train)

            if len(train) < 5:

                st.write('There is **_not enough sales data avaialble_** for this material ID')
            
            else:

                m = Prophet.Prophet(changepoint_prior_scale = 0.25)

                m.add_regressor('sales_volume', mode='multiplicative')

                m.fit(train)

                future = m.make_future_dataframe(periods=36,freq='MS')

                future.loc[:,'sales_volume'] = sales_forecast.loc[:,'yhat']

            #    future = future[(future['ds']< '2021-05-01')]

                future['floor'] = 0

                print(future)

                future = future.dropna()

                claims_forecast = m.predict(future)

                print(claims_forecast)

                df_final = df.merge(claims_forecast, how = 'outer', on = ['ds'])

                df_final['yhat'][df_final['yhat'] < 0] = 0

                # Plotting
                
                fig, ax = plt.subplots(figsize=(20,10))

                sns.lineplot(df_final.ds, df_final.y, label="actual", marker=11)
                sns.lineplot(df_final.ds[:(train_len)], df_final.y[:(train_len)], label="backtest", marker=11),
                sns.lineplot(df_final.ds[(train_len-1):], df_final.yhat[(train_len-1):], label="forecast", marker=11, linestyle="dashed")

                plt.xticks(rotation=90, fontsize=18)

                plt.xlabel('Repair date', fontsize=18)
                plt.ylabel('Claims count', fontsize=18)
                plt.title('Part number: 6740173201')
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                st.pyplot(fig)
        
        elif (len(sales_part_num) == 1) and (sales_part_num[0] != MATERIAL_SELECTED[0]):

            st.write('The claims part number is different from the sales part number - I am here!')

            cleansed_sales_df_filtered = cleansed_sales_df[cleansed_sales_df['Material'].str.contains(sales_part_num[0])]

            cleansed_sales_df_filtered = cleansed_sales_df_filtered.dropna(axis=1, how='all')

            cleansed_sales_df_filtered = pd.melt(cleansed_sales_df_filtered, id_vars=['Material','Product Name', 'M+H_Proj_#', 'M+H_Plant_2_Comp_Code','Date_Sales_Y/M'], var_name='date')
            
            merged_df = cleansed_sales_df_filtered.merge(claims_frequency_df_fcst, how = 'inner', on = ['date'])

            merged_df = merged_df.rename(columns={"value":"sales_volume"})

            #scaler = MinMaxScaler()

            #merged_df[['sales_volume', 'claims_count']] = scaler.fit_transform(merged_df[['sales_volume', 'claims_count']])

            merged_df = merged_df.drop(['Material', 'Product Name', 'M+H_Proj_#','Date_Sales_Y/M', 'M+H_Plant_2_Comp_Code'], axis=1)

            merged_df[['date']] = pd.to_datetime(merged_df['date'].astype(str))

            correlation = merged_df['sales_volume'].corr(merged_df['claims_count'])

            st.write('The correlation between sales and claims data is', correlation)

            print('merged_df', merged_df)

            df_for_forecast = merged_df.copy()

            #Generate a sales forecast first
            df_sales = pd.DataFrame()
            df_sales['ds'] = df_for_forecast['date']
            df_sales['y'] = df_for_forecast['sales_volume']

            df_sales = df_sales[(df_sales['ds'] > '2019-12-31') & (df_sales['ds'] < '2020-12-31')].reset_index(drop=True)

            print('df_sales', df_sales)
            
            m_sales = Prophet.Prophet(changepoint_prior_scale = 0.95)

            m_sales.fit(df_sales)

            future_sales = m_sales.make_future_dataframe(periods=12,freq='MS')

            print('future dataframe', future_sales)

            sales_forecast = m_sales.predict(future_sales)

            print('future sales', sales_forecast)

            # Now use the sales forecast as an input to the claims backtest/forecast
            df = pd.DataFrame()
            df['ds'] = df_for_forecast['date']
            df['y'] = df_for_forecast['claims_count']
            

            df = df.sort_values('ds').reset_index(drop = True)

            # For the 1354072S01 use case

            df = df[df['ds'] > '2019-12-31'].reset_index(drop = True)

            df['sales_volume'] = df_sales['y']

            len_df = len(df)

            print(len_df)

            print("Trimmed df for 1354072S01", df)

            train_len = int(0.75*len_df)

            future_len = len_df - train_len

            print(train_len, future_len)

            train = df.head(train_len).dropna()

            train = train.sort_values('ds').reset_index(drop = True)

            print('train', train)

            if len(train) < 5:

                st.write('There is **_not enough sales data avaialble_** for this material ID')
            
            else:

                m = Prophet.Prophet(changepoint_prior_scale = 0.25)

                m.add_regressor('sales_volume', mode='multiplicative')

                m.fit(train)

            #    future = m.make_future_dataframe(periods=future_len,freq='MS')
                future = m.make_future_dataframe(periods=12,freq='MS')

                future.loc[:,'sales_volume'] = sales_forecast.loc[:,'yhat']

               # future = future[(future['ds']< '2021-05-01')]

                future['floor'] = 0

                print('future dataframe', future)

                future = future.dropna()

                claims_forecast = m.predict(future)

                print('claims forecast', claims_forecast)

                df_final = df.merge(claims_forecast, how = 'outer', on = ['ds'])

                df_final['yhat'][df_final['yhat'] < 0] = 0

                # Plotting
                
                fig, ax = plt.subplots(figsize=(20,10))

                sns.lineplot(df_final.ds, df_final.y, label="actual", marker=11)
                sns.lineplot(df_final.ds[:(train_len)], df_final.y[:(train_len)], label="backtest", marker=11),
                sns.lineplot(df_final.ds[(train_len-1):], df_final.yhat[(train_len-1):], label="forecast", marker=11, linestyle="dashed")

                plt.xticks(rotation=90, fontsize=18)

                plt.xlabel('Repair date', fontsize=18)
                plt.ylabel('Claims count', fontsize=18)
                plt.title('Claims part number: 1354072S01 || Sales part number: 1222531S03')
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                st.pyplot(fig)
        
        elif (len(sales_part_num) > 1):

            st.write('We have multiple sales part numbers for a single claims part number!')

            cleansed_sales_df_filtered = cleansed_sales_df[(cleansed_sales_df['Material'].isin(sales_part_num))]

            print('cleansed_sales_df_filtered', cleansed_sales_df_filtered)

            cleansed_sales_df_filtered = cleansed_sales_df_filtered.dropna(axis=0, how='all')

            cleansed_sales_df_filtered = pd.melt(cleansed_sales_df_filtered, id_vars=['Material', 'Product Name', 'M+H_Proj_#', 'M+H_Plant_2_Comp_Code','Date_Sales_Y/M'], var_name='date')
            
            print('cleansed_sales_df_filtered', cleansed_sales_df_filtered)
            
            merged_df = cleansed_sales_df_filtered.merge(claims_frequency_df_fcst, how = 'inner', on = ['date'])

            merged_df = merged_df.rename(columns={"value":"sales_volume"})

            print('merged df before dropna', merged_df)

            merged_df = merged_df[merged_df['sales_volume'].notna()]

            merged_df = merged_df[merged_df['sales_volume'] != 0]

            #scaler = MinMaxScaler()

            #merged_df[['sales_volume', 'claims_count']] = scaler.fit_transform(merged_df[['sales_volume', 'claims_count']])

            merged_df = merged_df.drop(['Material', 'Product Name', 'M+H_Proj_#','Date_Sales_Y/M', 'M+H_Plant_2_Comp_Code'], axis=1)

            merged_df[['date']] = pd.to_datetime(merged_df['date'].astype(str))

            correlation = merged_df['sales_volume'].corr(merged_df['claims_count'])

            st.write('The correlation between sales and claims data is', correlation)

            print(merged_df)

            df_for_forecast = merged_df.copy()

            df = pd.DataFrame()
            df['ds'] = df_for_forecast['date']
            df['y'] = df_for_forecast['claims_count']
            df['sales_volume'] = df_for_forecast['sales_volume']

            df = df.sort_values('ds').reset_index(drop = True)

            len_df = len(df)

            print(len_df)

            train_len = int(0.75*len_df)

            future_len = len_df - train_len

            print(train_len, future_len)

            train = df.head(train_len).dropna()

            train = train.sort_values('ds').reset_index(drop = True)

            print(train)

            if len(train) < 5:

                st.write('There is **_not enough sales data avaialble_** for this material ID')
            
            else:

                m = Prophet.Prophet(changepoint_prior_scale = 0.05)

                m.add_regressor('sales_volume', mode='multiplicative')

                m.fit(train)

                future = m.make_future_dataframe(periods=future_len,freq='MS')

                future.loc[:,'sales_volume'] = df.loc[:,'sales_volume']

                future = future[(future['ds']< '2021-05-01')]

                future['floor'] = 0

                print(future)

                future = future.dropna()

                claims_forecast = m.predict(future)

                print(claims_forecast)

                df_final = df.merge(claims_forecast, how = 'inner', on = ['ds'])

                df_final['yhat'][df_final['yhat'] < 0] = 0

                # Plotting
                
                fig, ax = plt.subplots(figsize=(20,10))

                sns.lineplot(df_final.ds, df_final.y, label="actual", marker=11)
                sns.lineplot(df_final.ds[:(train_len)], df_final.y[:(train_len)], label="backtest", marker=11),
                sns.lineplot(df_final.ds[(train_len-1):], df_final.yhat[(train_len-1):], label="forecast", marker=11, linestyle="dashed")

                plt.xticks(rotation=90, fontsize=18)

                plt.xlabel('Repair date', fontsize=18)
                plt.ylabel('Count', fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                st.pyplot(fig)



    
    else: # elif include_sales_history == False & len(cleansed_sales_df_filtered) > 5

        merged_df = claims_frequency_df_fcst

        print(merged_df)

        merged_df[['date']] = pd.to_datetime(merged_df['date'].astype(str)) 

        df_for_forecast = merged_df.copy()

        df = pd.DataFrame()
        df['ds'] = df_for_forecast['date']
        df['y'] = df_for_forecast['claims_count']

        df = df.sort_values('ds').reset_index(drop = True)

        len_df = len(df)

        print(len_df)

        train_len = int(0.7*len_df)

        future_len = len_df - train_len

        print(train_len, future_len)

        train = df.head(train_len).dropna()

        train = train.sort_values('ds').reset_index(drop = True)

        print(train)

        m = Prophet.Prophet(changepoint_prior_scale = 0.25, seasonality_prior_scale = 0.1)

        m.fit(train)

        future = m.make_future_dataframe(periods=future_len,freq='MS')

        future = future[(future['ds']< '2021-05-01')]

        future['floor'] = 0

        print(future)

        claims_forecast = m.predict(future)

        print(claims_forecast)

        df_final = df.merge(claims_forecast, how = 'inner', on = ['ds'])

        df_final['yhat'][df_final['yhat'] < 0] = 0

        # Plotting
        
        fig, ax = plt.subplots(figsize=(20,10))

        sns.lineplot(df_final.ds, df_final.y, label="actual", marker=11)
        sns.lineplot(df_final.ds[:(train_len)], df_final.y[:(train_len)], label="backtest", marker=11),
        sns.lineplot(df_final.ds[(train_len-1):], df_final.yhat[(train_len-1):], label="forecast", marker=11, linestyle="dashed")

        plt.xticks(rotation=90, fontsize=18)

        plt.xlabel('Repair date', fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        st.pyplot(fig)

    # elif len(cleansed_sales_df_filtered) < 5 & include_sales_history == True:

    #     st.write('There is **_not enough sales data avaialble_** for this material ID')



elif warranty_data_plotting == 'Forecast':


    merged_df = claims_frequency_df_fcst

    print(merged_df)

    merged_df[['date']] = pd.to_datetime(merged_df['date'].astype(str)) 

    df_for_forecast = merged_df.copy()

    df = pd.DataFrame()
    df['ds'] = df_for_forecast['date']
    df['y'] = df_for_forecast['claims_count']

    df = df.sort_values('ds').reset_index(drop = True)

    m = Prophet.Prophet(changepoint_prior_scale = 0.05, seasonality_prior_scale = 0.1)

    m.fit(df)

    future = m.make_future_dataframe(periods=12,freq='MS')

    future['floor'] = 0

    print(future)

    claims_forecast = m.predict(future)

    print(claims_forecast)

    claims_forecast['yhat'][claims_forecast['yhat'] < 0] = 0

    # Plotting
    
    fig, ax = plt.subplots(figsize=(20,10))

    sns.lineplot(df.ds, df.y, label="actual", marker=11)
    sns.lineplot(claims_forecast.ds[-13:], claims_forecast.yhat[-13:], label="forecast", marker=11, linestyle="dashed")

    plt.xticks(rotation=90, fontsize=18)

    plt.xlabel('Repair date', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    st.pyplot(fig)

