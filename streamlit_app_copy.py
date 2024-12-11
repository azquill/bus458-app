import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import re
import pandas as pd

# Load the model
with open("/workspaces/bus458-app/data_science_salary.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

# Function to make the salary prediction
def sal_predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

#Mapping for degree
education_mapping = {
    "No formal education past high school": 0,
    "Some college/university study without earning a bachelor’s degree": 2,
    "Bachelor’s degree": 3,
    "Master’s degree": 4,
    "Doctoral degree": 5,
    "Professional doctorate": 6,
    "I prefer not to answer": 1
}

#ml_experience_mapping
ml_experience_mapping = {
    "I do not use machine learning methods": 0,
    "Under 1 year": 1,
    "1-2 years": 2,
    "2-3 years": 3,
    "3-4 years": 4,
    "4-5 years": 5,
    "5-10 years": 6,
    "10-20 years": 7,
    "20 or more years": 8
}

# Define mappings for encoding
country_mapping = {name: idx for idx, name in enumerate([
    "Algeria", "Argentina", "Australia", "Bangladesh", "Belgium", "Brazil",
    "Cameroon", "Canada", "Chile", "China", "Colombia", "Czech Republic",
    "Egypt", "Ecuador", "Ethiopia", "France", "Germany", "Ghana",
    "Hong Kong (S.A.R.)", "India", "Indonesia", "Iran, Islamic Republic of...",
    "Ireland", "Israel", "Italy", "Japan", "Kenya", "Malaysia",
    "Mexico", "Morocco", "Nepal", "Netherlands", "Nigeria", "Pakistan",
    "Peru", "Philippines", "Poland", "Portugal", "Romania", "Russia",
    "Saudi Arabia", "Singapore", "South Africa", "South Korea", "Spain",
    "Sri Lanka", "Taiwan", "Thailand", "Tunisia", "Turkey", "Ukraine",
    "United Arab Emirates", "United Kingdom of Great Britain and Northern Ireland",
    "United States of America", "Viet Nam", "Zimbabwe", "Other",
    "I do not wish to disclose my location"
])}

country_one_hot_columns = [
    'Q4_Argentina', 'Q4_Australia', 'Q4_Algeria', 'Q4_Bangladesh', 'Q4_Belgium', 'Q4_Brazil', 'Q4_Cameroon', 
    'Q4_Canada', 'Q4_Chile', 'Q4_China', 'Q4_Colombia', 'Q4_Czech Republic', 'Q4_Ecuador', 'Q4_Egypt', 
    'Q4_Ethiopia', 'Q4_France', 'Q4_Germany', 'Q4_Ghana', 'Q4_Hong Kong (S.A.R.)', 'Q4_India', 'Q4_Indonesia', 
    'Q4_Iran, Islamic Republic of...', 'Q4_Ireland', 'Q4_Israel', 'Q4_Italy', 'Q4_Japan', 'Q4_Kenya', 
    'Q4_Malaysia', 'Q4_Mexico', 'Q4_Morocco', 'Q4_Nepal', 'Q4_Netherlands', 'Q4_Nigeria', 'Q4_Other', 
    'Q4_Pakistan', 'Q4_Peru', 'Q4_Philippines', 'Q4_Poland', 'Q4_Portugal', 'Q4_Romania', 'Q4_Russia', 
    'Q4_Saudi Arabia', 'Q4_Singapore', 'Q4_South Africa', 'Q4_South Korea', 'Q4_Spain', 'Q4_Sri Lanka', 
    'Q4_Taiwan', 'Q4_Thailand', 'Q4_Tunisia', 'Q4_Turkey', 'Q4_Ukraine', 'Q4_United Arab Emirates', 
    'Q4_United Kingdom of Great Britain and Northern Ireland', 'Q4_United States of America', 
    'Q4_Viet Nam', 'Q4_Zimbabwe', 'Q4_I do not wish to disclose my location'
]


# Create the mapping from country to one-hot encoded column name
country_one_hot_mapping = {
    'Argentina': 'Q4_Argentina', 
    'Algeria': 'Q4_Algeria',
    'Australia': 'Q4_Australia',
    'Bangladesh': 'Q4_Bangladesh',
    'Belgium': 'Q4_Belgium',
    'Brazil': 'Q4_Brazil',
    'Cameroon': 'Q4_Cameroon',
    'Canada': 'Q4_Canada',
    'Chile': 'Q4_Chile',
    'China': 'Q4_China',
    'Colombia': 'Q4_Colombia',
    'Czech Republic': 'Q4_Czech Republic',
    'Ecuador': 'Q4_Ecuador',
    'Egypt': 'Q4_Egypt',
    'Ethiopia': 'Q4_Ethiopia',
    'France': 'Q4_France',
    'Germany': 'Q4_Germany',
    'Ghana': 'Q4_Ghana',
    'Hong Kong (S.A.R.)': 'Q4_Hong Kong (S.A.R.)',
    'India': 'Q4_India',
    'Indonesia': 'Q4_Indonesia',
    'Iran, Islamic Republic of...': 'Q4_Iran, Islamic Republic of...',
    'Ireland': 'Q4_Ireland',
    'Israel': 'Q4_Israel',
    'Italy': 'Q4_Italy',
    'Japan': 'Q4_Japan',
    'Kenya': 'Q4_Kenya',
    'Malaysia': 'Q4_Malaysia',
    'Mexico': 'Q4_Mexico',
    'Morocco': 'Q4_Morocco',
    'Nepal': 'Q4_Nepal',
    'Netherlands': 'Q4_Netherlands',
    'Nigeria': 'Q4_Nigeria',
    'Other': 'Q4_Other',
    'Pakistan': 'Q4_Pakistan',
    'Peru': 'Q4_Peru',
    'Philippines': 'Q4_Philippines',
    'Poland': 'Q4_Poland',
    'Portugal': 'Q4_Portugal',
    'Romania': 'Q4_Romania',
    'Russia': 'Q4_Russia',
    'Saudi Arabia': 'Q4_Saudi Arabia',
    'Singapore': 'Q4_Singapore',
    'South Africa': 'Q4_South Africa',
    'South Korea': 'Q4_South Korea',
    'Spain': 'Q4_Spain',
    'Sri Lanka': 'Q4_Sri Lanka',
    'Taiwan': 'Q4_Taiwan',
    'Thailand': 'Q4_Thailand',
    'Tunisia': 'Q4_Tunisia',
    'Turkey': 'Q4_Turkey',
    'Ukraine': 'Q4_Ukraine',
    'United Arab Emirates': 'Q4_United Arab Emirates',
    'United Kingdom of Great Britain and Northern Ireland': 'Q4_United Kingdom of Great Britain and Northern Ireland',
    'United States of America': 'Q4_United States of America',
    'Viet Nam': 'Q4_Viet Nam',
    'Zimbabwe': 'Q4_Zimbabwe',
    'I do not wish to disclose my location': 'Q4_I do not wish to disclose my location',
}

years_of_coding_mapping = {
    "I have never written code": 0,
    "< 1 years": 1,
    "1-2 years": 2,
    "3-5 years": 3,
    "5-10 years": 4,
    "10-20 years": 5,
    "20+ years": 6
}

ml_spending_mapping = {
    "$0 ($USD)": 0,
    "$1-$99": 1,
    "$100-$999": 2,
    "$1000-$9,999": 3,
    "$10,000-$99,999": 4,
    "$100,000 or more ($USD)": 5
}

ml_usage_mapping = {
    "We are exploring ML methods (and may one day put a model into production)": 0,
    "We use ML methods for generating insights (but do not put working models into production)": 1,
    "We recently started using ML methods (i.e., models in production for less than 2 years)": 2,
    "We have well established ML methods (i.e., models in production for more than 2 years)": 3,
    "No (we do not use ML methods)": 4,
    "I do not know": 5
}

ml_company_mapping = {
    "We are exploring ML methods (and may one day put a model into production)": 2,
    "We use ML methods for generating insights (but do not put working models into production)": 3, 
    "We recently started using ML methods (i.e. models in production for less than 2 years)": 4, 
    "We have well established ML methods (i.e. models in production for more than 2 years)": 5, 
    "No (we do not use ML methods)": 0, 
    "I do not know": 1
}

def process_ranges(value):
    if isinstance(value, str):
        if "I have never written code" in value or "I do not use machine learning methods" in value:
            return 0
        value = re.sub(r'[^0-9\-]', '', value)
        parts = value.split('-')
        if len(parts) == 2:
            return (int(parts[0]) + int(parts[1])) // 2
        elif len(parts) == 1 and parts[0]:
            return int(parts[0])
    return None

# Main function to set up the Streamlit interface
def main():
    st.title('Salary Predictor for Data Scientists')
    st.subheader('Complete the following 5 questions to see your current salary prediction based off your skillset.')

    # Dropdowns for each feature
    q2_value = st.number_input('Age', min_value=0, max_value=120)
    q16_value = st.selectbox('Years using ML methods?', options=(ml_experience_mapping.keys()))
    q4_value = st.selectbox('Country', options=list(country_mapping.keys()))
    q27_value = st.selectbox('Does your current employer incorporate machine learning methods into their business?', options=list(ml_usage_mapping.keys()))
    q30_value = st.selectbox('Money Spent on Machine Learning', options=list(ml_spending_mapping.keys()))
    q11_value = st.selectbox('For how many years have you been writing code and/or programming?', options=list(years_of_coding_mapping.keys()))
    q8_value = st.selectbox('Education', options=list(education_mapping.keys()))

    # Encode country using one-hot encoding by setting 1 for selected country, 0 for others
    country_one_hot = {col: 0 for col in country_one_hot_mapping.values()}  # Start with all zeros
    country_one_hot[country_one_hot_mapping[q4_value]] = 1  # Set selected country to 1

    # Encode country using one-hot encoding by setting 1 for selected country, 0 for others
    country_one_hot = {col: 0 for col in country_one_hot_columns}  # Start with all zeros
    country_one_hot[country_one_hot_mapping[q4_value]] = 1  # Set selected country to 1

    # Flatten the one-hot encoding into a list
    country_encoded = list(country_one_hot.values())


    # Convert selected values to numerical format
    q16_value_encoded = process_ranges(q16_value)
    q27_value_encoded = 0 if q27_value == "No (we do not use ML methods)" else 1
    q30_value_encoded = process_ranges(q30_value)
    q11_value_encoded = process_ranges(q11_value)
    q8_value_encoded = education_mapping[q8_value]
    
    # Create a dictionary to match the encoded values with column names
    input_data_dict = {
        "Q2": q2_value,
        "Q16": q16_value_encoded,
        "Q4": country_encoded,
        "Q27": q27_value_encoded,
        "Q30": q30_value_encoded,
        "Q11": q11_value_encoded,
        "Q8": q8_value_encoded,
    }

    # Prepare input data for the model
    input_data = [
        input_data_dict["Q27"],
        input_data_dict["Q30"],
        *input_data_dict["Q4"],
        input_data_dict["Q2"],
        input_data_dict["Q11"],
        input_data_dict["Q8"],
        input_data_dict["Q16"],
    ]

    st.write(input_data)

    # Predict when the button is pressed
    if st.button('Predict Salary'):
        try:
            salary = sal_predict(input_data)
            st.success(f"Predicted Salary: ${salary:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()