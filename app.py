# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import matplotlib.ticker as mticker

from scipy.stats import zscore
from zipfile import ZipFile
from io import BytesIO
from collections import Counter
from duckduckgo_search import DDGS
    


# Function to load the datasets as a mergered dataset
@st.cache_data  # Cache the function to enhance performance
def load_data():
    # Defining the file paths
    url_0 = "https://raw.githubusercontent.com/aaubs/ds-master/main/data/assignments_datasets/KIVA/kiva_loans_part_0.csv.zip"
    df_0 = pd.read_csv(ZipFile(BytesIO(requests.get(url_0).content)).open("kiva_loans_part_0.csv"))

    url_1 = "https://raw.githubusercontent.com/aaubs/ds-master/main/data/assignments_datasets/KIVA/kiva_loans_part_1.csv.zip"
    df_1 = pd.read_csv(ZipFile(BytesIO(requests.get(url_1).content)).open("kiva_loans_part_1.csv"))

    url_2 = "https://raw.githubusercontent.com/aaubs/ds-master/main/data/assignments_datasets/KIVA/kiva_loans_part_2.csv.zip"
    df_2 = pd.read_csv(ZipFile(BytesIO(requests.get(url_2).content)).open("kiva_loans_part_2.csv"))

    #Mergering data
    df_merged = pd.concat([df_0, df_1, df_2])
    # Dropping unused columns
    df = df_merged.drop(['id', 'use', 'partner_id', 'posted_time', 'disbursed_time', 'funded_time', 'tags'], axis=1)
    # Filling missing regions with 'Unknown' and dropping rows with missing Borrower Gender & Country Code
    df['region'] = df['region'].fillna('Unknown')
    df = df.dropna(subset=['borrower_genders', 'country_code'])


    #Using ChatGPT to help cleaning out the wierd gender observations
    def clean_gender(gender_string):
        # Splits the string by commas to get a list of genders
        genders = gender_string.split(', ')

        # Counts the frequency of each gender
        gender_count = Counter(genders)

        # Returns the gender with the highest frequency
        return gender_count.most_common(1)[0][0]

    # Applying the function to the 'borrower_genders' column
    df['borrower_genders'] = df['borrower_genders'].apply(clean_gender)

    z_scores_loan = zscore(df['loan_amount'])
    z_scores_term = zscore(df['term_in_months'])
    z_scores_lender = zscore(df['lender_count'])

    # Identifying outliers where Z-scores are greater than 3 or less than -3 to get 95% of data within 2 standard deviations
    df['outlier_loan_amount'] = (z_scores_loan > 3) | (z_scores_loan < -3)
    df['outlier_term_in_months'] = (z_scores_term > 3) | (z_scores_term < -3)
    df['outlier_lender_count'] = (z_scores_lender > 3) | (z_scores_lender < -3)

    # Combine outliers from all columns
    df['outlier'] = df['outlier_loan_amount'] | df['outlier_term_in_months'] | df['outlier_lender_count']

    # Remove the rows with outliers
    df = df[~df['outlier']]

    # Drop the helper columns used for outlier identification
    df = df.drop(columns=['outlier', 'outlier_loan_amount', 'outlier_term_in_months', 'outlier_lender_count'])

    # Replace 'female' with 'Female' and 'male' with 'Male'
    df['borrower_genders'] = df['borrower_genders'].replace({'female': 'Female', 'male': 'Male'})

    # Defining income level groups
    income_groups = {
    'Low': ['Somalia', 'Yemen', 'Burundi', 'Sierra Leone', 'Timor-Leste', 'Mozambique', 'Burkina Faso', 'Togo', 'Benin', 'Vanuatu', 'Saint Vincent and the Grenadines', 'Malawi', 'South Sudan', 'Rwanda', 'Zambia', 'Madagascar', 'Afghanistan', 'Nepal', 'Palestine', 'Haiti', 'Democratic Republic of the Congo', 'Central African Republic', 'Guam', 'Bolivia','Samoa','Uganda','Kosovo','Azerbaijan','Nigeria','Solomon Islands','Solomon Islands','Belize','Suriname',"Lao People's Democratic Republic",'Myanmar (Burma)','Kungyangon','Lesotho'],
    'Lower Middle': ['Pakistan','India', 'Vietnam', 'Indonesia', 'Peru', 'Kenya', 'Ghana', 'Mali', 'Nicaragua', 'Honduras', 'Guatemala', 'Ecuador', 'Tajikistan', 'Armenia', 'Paraguay', 'Kyrgyzstan', 'Cambodia', 'El Salvador', 'Liberia', 'Virgin Islands', 'Tanzania','Senegal','Mongolia','Cameroon','The Democratic Republic of the Congo','Zimbabwe','Dominican Republic','Moldova','Congo'],
    'Upper Middle': ['Brazil', 'China', 'Mexico', 'Jordan', 'Philippines', 'Thailand', 'South Africa', 'Lebanon', 'Albania', 'Georgia', 'Colombia', 'Ukraine', 'Puerto Rico', 'Iraq','Egypt'],
    'High': ['United States', 'Israel', 'Turkey', 'Costa Rica', 'Panama','Chile']
    }

    # Creating a dictionary mapping countries to income levels
    country_to_income = {country: level for level, countries in income_groups.items() for country in countries}

    # Add a new column 'income_level' based on the 'country' column
    df['income_level'] = df['country'].map(country_to_income)

    return df

# Load the data using the defined function
df = load_data()

# Set the app title and sidebar header
st.title("Kiva Loan Analytics Dashboard 💸")
st.sidebar.header("Filters 📊")

# Introduction

# HR Attrition Dashboard

st.markdown("""
            Welcome to the Kiva Loan Dashboard. As global efforts to alleviate poverty through microfinance continue to grow, understanding the factors that influence loan disbursements and repayments is crucial. This interactive dashboard leverages data analytics to provide insights into borrower demographics, loan amounts, and regional disparities. Through this lens, we aim to reveal the underlying trends in microloans and provide actionable strategies to enhance Kiva's impact in underserved communities worldwide.
""")
with st.expander("📊 **Objective**"):
                 st.markdown("""
At the core of this dashboard is the mission to visually analyze Kiva’s loan data, empowering stakeholders with insights to answer critical questions such as:
- Which regions and borrower demographics are most in need of financial support?
- What factors influence loan amounts and loan performance across different sectors?
- Based on observed trends, what strategies can be implemented to optimize loan allocation and increase Kiva's impact in alleviating poverty?
"""
)
                             
# Tutorial Expander
with st.expander("How to Use the Dashboard 📚"):
    st.markdown("""
    1. **Filter Data** - Use the sidebar filters to narrow down specific data sets.
    2. **Visualize Data** - From the dropdown, select a visualization type to view patterns.
    3. **Insights & Recommendations** - Scroll down to see insights derived from the visualizations and actionable recommendations.
    """)


# Sidebar filter: Borrower Gender
selected_borrower_gender = st.sidebar.multiselect("Select Borrower Gender 🕰️", df['borrower_genders'].unique().tolist(), default=df['borrower_genders'].unique().tolist())
if not selected_borrower_gender:
    st.warning("Please select the borrower gender(s) from the sidebar ⚠️")
    st.stop()
filtered_df = df[df['borrower_genders'].isin(selected_borrower_gender)]

# Sidebar filter: Income Levels 
income_levels = df['income_level'].unique().tolist()
selected_income_levels = st.sidebar.multiselect("Select Country Income Level 🏢", income_levels, default=income_levels)
if not selected_income_levels:
    st.warning("Please select the Income Level(s) from the sidebar ⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['income_level'].isin(selected_income_levels)]

# Sidebar filter: Sectors
sectors = df['sector'].unique().tolist()
selected_sectors = st.sidebar.multiselect("Select Sectors 🏢", sectors, default=sectors)
if not selected_sectors:
    st.warning("Please select the sector(s) from the sidebar ⚠️")
    st.stop()
filtered_df = filtered_df[filtered_df['sector'].isin(selected_sectors)]

# Sidebar filter: Loan Amount Range
min_loan_amount = int(df['loan_amount'].min())
max_loan_amount = int(df['loan_amount'].max())
loan_amount_range = st.sidebar.slider("Select Loan Amount Range 💰", min_loan_amount, max_loan_amount, (min_loan_amount, max_loan_amount))
filtered_df = filtered_df[(filtered_df['loan_amount'] >= loan_amount_range[0]) & (filtered_df['loan_amount'] <= loan_amount_range[1])]

# Dropdown to select the type of visualization
visualization_option = st.selectbox(
    "Select Visualization 🎨", 
    ["Distribution of Loan Amount (Boxplot)",
     "Loan Amount by Borrower Gender", 
     "KDE Plot: Term in Months by Gender", 
     "Sector by Borrower Gender", 
     "Average Loan Amount by Income Level and Borrower Gender", 
     "Correlation Heatmap"]
)

if visualization_option == "Distribution of Loan Amount (Boxplot)":
    st.header("Distribution of Loan Amount (Boxplot)")
    st.markdown("""This section visualizes the distribution of loan amounts using a boxplot. 
    It shows how loan amounts are spread across the data and helps identify outliers and the range of loans provided.""")

    # Create an Altair boxplot with adjusted width
    boxplot = alt.Chart(filtered_df).mark_boxplot(size=300).encode(  # Adjust 'size' to control the width of the boxes
    y=alt.Y('loan_amount:Q', title='Loan Amount')  # Quantitative variable on y-axis with a label
    ).properties(
    title='Distribution of Loan Amounts (Boxplot)',
    width=200  # Adjust the width of the entire plot if needed
    )

    # Display the boxplot in Streamlit
    st.altair_chart(boxplot, use_container_width=True)

    # AI Button
    if st.button('Please explain the boxplot results'):
     with st.expander("AI Explanation"):
        boxplot_data = filtered_df[['loan_amount']].describe()
        st.markdown(DDGS().chat("You are a very intelligent Data Analyst: Provide a concise and precise explanation of the boxplot showing the distribution of"+str(boxplot_data), model='gpt-4o-mini'))

# Visualizations based on user selection
elif visualization_option == "Loan Amount by Borrower Gender":
    st.header("Loan Amount by Borrower Gender")
    st.markdown("""This section presents a bar chart showing the loan amounts distributed by borrower gender. 
    You can explore the distribution and see how loan amounts differ between genders.""")

    # Calculating the maximum loan amount in the filtered DataFrame to be able to make the chart x-axis dynamic to match sidebar filter
    max_loan_amount = filtered_df['loan_amount'].max()
    # Bar chart for Loan Amount by Borrower Gender
    chart = alt.Chart(filtered_df).mark_bar().encode(
        x=alt.X('loan_amount', scale=alt.Scale(domain=(0, max_loan_amount)), title='Loan Amount'), 
        y='count()',
        color=alt.Color('borrower_genders', title='Borrower Gender')
    ).properties(
        title='Borrower Gender by Loan Amount'
    )
    #Display altair chart
    st.altair_chart(chart, use_container_width=True)

    #AI Button
    if st.button('Please explain the results of this chart'):
     with st.expander("AI Explanation"):
        loan_amount_data = filtered_df.groupby(['loan_amount', 'borrower_genders']).size().reset_index(name='count')
        st.markdown(DDGS().chat("You are a very intelligent Data Analyst: Provide a concise and precise explanation of the results from"+str(loan_amount_data), model='gpt-4o-mini'))
    
elif visualization_option == "KDE Plot: Term in Months by Gender":
    st.header("Term in Months by Gender (KDE Plot)")
    st.markdown("""This section shows the distribution of loan terms (in months) by gender using a Kernel Density Estimate (KDE) plot. 
    The KDE plot helps understand the most common loan terms across different genders.""")

    # KDE plot for Term in Months based on Gender
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=filtered_df, x='term_in_months', hue='borrower_genders', fill=True, palette='Set2')
    plt.xlabel('Term in Months')
    plt.ylabel('Density')   
    plt.title('KDE Plot of Term in Months by Borrower Gender')
    st.pyplot(plt)
    
    #AI Button
    if st.button('Please explain the chart results'):
     with st.expander("AI Explanation"):
        kde_data = filtered_df.groupby(['term_in_months', 'borrower_genders']).size().reset_index(name='count')
        st.markdown(DDGS().chat("You are a very intelligent Data Analyst: Provide a concise and precise explanation of the results from"+str(kde_data), model='gpt-4o-mini'))
        

elif visualization_option == "Sector by Borrower Gender":
    st.header("Sector by Borrower Gender")
    st.markdown("""This section visualizes the count of borrowers by sector, with colors representing gender. 
    It helps identify which sectors have a higher participation of male or female borrowers.""")

    # Bar chart for Sector by Borrower Gender
    chart = alt.Chart(filtered_df).mark_bar().encode(
        y=alt.Y('sector:O', title='Sector'),
        x=alt.X('count():Q'),
        color=alt.Color('borrower_genders:N', title='Borrower Genders')).properties(title='Sector by Borrower Gender')
    
    #Display altair chart
    st.altair_chart(chart, use_container_width=True)

    #AI Button
    if st.button('Please explain the chart results'):
     with st.expander("AI Explanation"):
        sector_data = filtered_df.groupby(['sector', 'borrower_genders']).size().reset_index(name='count')
        st.markdown(DDGS().chat("You are a very intelligent Data Analyst: Provide a concise and precise explanation of the results from"+str(sector_data), model='gpt-4o-mini'))


elif visualization_option == 'Average Loan Amount by Income Level and Borrower Gender':
    st.header("Average Loan Amount by Income Level and Borrower Gender")
    st.markdown("""This section displays a bar chart showing the average loan amount distributed by income level and gender. 
    It allows comparison of loan amounts across different income groups and borrower genders.""")

    # Creating an Altair bar chart with color based on 'borrower_genders'
    aggregated_data = filtered_df.groupby(['income_level', 'borrower_genders'])['loan_amount'].mean().reset_index()
    bar_chart = alt.Chart(aggregated_data).mark_bar().encode(
    x=alt.X('income_level:N', title='Income Level'),
    y=alt.Y('loan_amount:Q', title='Average Loan Amount'),
    color=alt.Color('borrower_genders:N', title='Borrower Gender'),
    tooltip=['income_level:N', 'borrower_genders:N', 'loan_amount:Q']).properties(title='Average Loan Amount by Income Level and Borrower Gender', width=600)
    # Displaying altair chart
    st.altair_chart(bar_chart, use_container_width=True)

    #AI Button
    if st.button('Please explain the bar chart results'):
     with st.expander("AI Explanation"):
          st.markdown(DDGS().chat("You are a very intelligent Data Analysist: Provide an a concise and precise expanation of the results from "+str(filtered_df.groupby(['income_level', 'borrower_genders'])['loan_amount'].mean().reset_index()), model='gpt-4o-mini'))

elif visualization_option == "Correlation Heatmap":
    st.header("Correlation Heatmap")
    st.markdown("""This section shows a correlation heatmap of selected Kiva loan attributes, such as loan amount, funded amount, lender count, and loan term. 
    It helps identify the relationships between these variables and understand how they are correlated.""")

    # Creating a heatmap based on selected columns for the correlation matrix from filtered data
    columns = ['loan_amount', 'funded_amount', 'lender_count', 'term_in_months']
    filtered_data = filtered_df[columns]
    # Creating a correlation matrix from the filtered data
    corr_matrix = filtered_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax).set_title('Correlation Matrix of KIVA Attributes')
    # Display the figure with the heatmap in Streamlit
    st.pyplot(fig)

    #AI Button
    if st.button('Please explain the heatmap results'):
     with st.expander("AI Explanation"):
          st.markdown(DDGS().chat("You are a very intelligent Data Analysist: Provide an a concise and precise expanation of the results from "+str(filtered_data.corr()), model='gpt-4o-mini'))
 

# Display dataset overview
st.header("Dataset Overview")
st.dataframe(df.describe())

if st.button('Please explain this dataset'):
     with st.expander("AI Explanation"):
          st.markdown(DDGS().chat("You are a very intelligent Data Analysist: Provide an a concise and precise expanation of the results from "+str(df.describe()), model='gpt-4o-mini'))


# Footer
st.markdown("""
    ---
    **Kiva Loans Impact**: Kiva is a non-profit organization that allows people to lend money to low-income entrepreneurs and students around the world. Through these loans, Kiva helps empower individuals, improve livelihoods, and reduce poverty.
""")    