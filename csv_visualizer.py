import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="B2Twin Data Visualization", layout="wide")

# Title and description
st.title("Biosphere 2 Data Visualization Tool")
st.write("Upload your CSV file from Biosphere 2 to visualize the data")

# File uploader
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Read and concatenate all CSV files
    dfs = {}  # Use dictionary to store dataframes by filename
    for uploaded_file in uploaded_files:
        df_temp = pd.read_csv(uploaded_file)
        # Convert timestamp column to datetime if it exists
        date_columns = df_temp.filter(like='Date').columns
        if len(date_columns) > 0:
            df_temp[date_columns[0]] = pd.to_datetime(df_temp[date_columns[0]])
            df_temp.set_index(date_columns[0], inplace=True)
        
        # Store dataframe with filename as key
        dfs[uploaded_file.name] = df_temp
    
    # Merge selected dataframes on index (timestamp)
    selected_files = st.multiselect(
        "Select files to merge and visualize",
        options=list(dfs.keys()),
        default=list(dfs.keys())
    )
    
    if selected_files:
        # Merge selected dataframes with suffixes
        df_merged = dfs[selected_files[0]]
        for filename in selected_files[1:]:
            df_merged = df_merged.join(dfs[filename], 
                                     rsuffix=f'_{filename}')
        
        # Reset index to make the date column visible
        df_merged.reset_index(inplace=True)
        
        # Display basic information about the merged dataset
        st.subheader("Dataset Overview")
        st.write(f"Number of files merged: {len(selected_files)}")
        st.write(f"Total number of rows: {df_merged.shape[0]}")
        st.write(f"Number of columns: {df_merged.shape[1]}")
        
        # Show the first few rows of the merged data
        st.subheader("Preview of the Merged Data")
        st.dataframe(df_merged.head())
        
        # Select columns for visualization
        st.subheader("Data Visualization")
        numeric_columns = df_merged.select_dtypes(include=['float64', 'int64']).columns
        
        # Create different types of plots based on the data
        plot_type = st.selectbox("Select Plot Type", 
                                ["Line Plot", "Scatter Plot", "Box Plot", "Histogram"])
        
        if plot_type == "Line Plot":
            # Get the date column (it should be the first column after reset_index)
            date_column = df_merged.columns[0]  # This should be your timestamp column
            y_column = st.selectbox("Select Y-axis data", numeric_columns)
            fig = px.line(df_merged, x=date_column, y=y_column, 
                         title=f'{y_column} Over Time')
            st.plotly_chart(fig)
            
        elif plot_type == "Scatter Plot":
            x_column = st.selectbox("Select X-axis data", numeric_columns)
            y_column = st.selectbox("Select Y-axis data", numeric_columns)
            fig = px.scatter(df_merged, x=x_column, y=y_column, 
                           title=f'{y_column} vs {x_column}')
            st.plotly_chart(fig)
            
        elif plot_type == "Box Plot":
            column = st.selectbox("Select data to plot", numeric_columns)
            fig = px.box(df_merged, y=column, title=f'Box Plot of {column}')
            st.plotly_chart(fig)
            
        elif plot_type == "Histogram":
            column = st.selectbox("Select data to plot", numeric_columns)
            fig = px.histogram(df_merged, x=column, title=f'Histogram of {column}')
            st.plotly_chart(fig)
        
        # Display basic statistics
        st.subheader("Statistical Summary")
        st.write(df_merged.describe())
