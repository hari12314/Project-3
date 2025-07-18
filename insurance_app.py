import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# Suppress widget-related warnings
warnings.filterwarnings("ignore", message="Error displaying widget")

# Load Pickle file
model = joblib.load("best_model.pkl")

# Streamlit App
st.set_page_config(page_title="Medical Insurance Estimator", layout="wide")
st.title(" Medical Insurance Cost Estimator")

menu = st.sidebar.radio("Navigation", ["Introduction", "EDA Insights", "Prediction"])

# Introduction
if menu == "Introduction":
    st.markdown("""
    ### Project Overview
    This app predicts **medical insurance charges** based on age, gender, BMI, smoker status, and region.

    **Use Cases:**
    - Insurance companies can personalize premiums.
    - Individuals can plan or compare insurance costs.
    - Healthcare consultants can estimate out-of-pocket expenses.
    """)

# EDA Insights
elif menu == "EDA Insights":
    st.markdown("###  Exploratory Data Analysis (EDA)")
    df = pd.read_csv("medical_insurance.csv")

    analysis_type = st.selectbox("Select Analysis Type", ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Outlier Detection", "Correlation Analysis"])

    if analysis_type == "Univariate Analysis":
        feature = st.selectbox("Choose a feature", ["What is the distribution of medical insurance charges?", "What is the age distribution of the individuals?", "How many people are smokers vs non-smokers?", "What is the average BMI in the dataset?", "Which regions have the most number of policyholders?"])

        if feature == "What is the distribution of medical insurance charges?":
            st.subheader("Distribution of Charges")
            fig, ax = plt.subplots()
            sns.histplot(df['charges'], bins=30, kde=True, ax=ax, color='maroon')
            ax.set_title("Distribution of Medical Insurance Charges")
            ax.set_xlabel("Charges")
            ax.set_ylabel("Frequency")
            ax.grid(True)
            st.pyplot(fig)

        elif feature == "What is the age distribution of the individuals?":
            st.subheader("Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['age'], bins=30, kde=True, ax=ax, color='skyblue')
            ax.set_title("Age Distribution of Individuals")
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif feature == "How many people are smokers vs non-smokers?":
            st.subheader("Smoker Count")
            fig, ax = plt.subplots()
            sns.countplot(x='smoker', data=df, hue='smoker', legend=False, palette='Set2', ax=ax)
            ax.set_title("Smokers vs Non-Smokers")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        elif feature == "What is the average BMI in the dataset?":
            st.subheader("BMI Distribution")
            mean_bmi = df['bmi'].mean()
            st.write(f" Average BMI: **{mean_bmi:.2f}**")
            
            fig, ax = plt.subplots()
            sns.boxplot(x=df['bmi'], color='lightblue', ax=ax)
            ax.set_title("Distribution of BMI")
            ax.set_xlabel("BMI")
            ax.legend()
            st.pyplot(fig)

        elif feature == "Which regions have the most number of policyholders?":
            st.subheader("Number of Policyholders by Region")
            fig, ax = plt.subplots()
            sns.countplot(x='region', data=df, order=df['region'].value_counts().index, hue='region', legend=False, palette='pastel', ax=ax)
            ax.set_title("Policyholders in Each Region")
            ax.set_xlabel("Region")
            ax.set_ylabel("Count")
            st.pyplot(fig)

    elif analysis_type == "Bivariate Analysis":
        feature_pair = st.selectbox("Choose analysis", [
            "How do charges vary with age?", 
            "Is there a difference in average charges between smokers and non-smokers?", 
            "Does BMI impact insurance charges?", 
            "Do men or women pay more on average?", 
            "Is there a correlation between the number of children and the insurance charges?"])

        if feature_pair == "How do charges vary with age?":
            fig, ax = plt.subplots()
             # sns.scatterplot(x='age', y='charges', data=df, ax=ax)
            sns.regplot(x='age', y='charges', data=df, scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'})
            ax.set_title("Charges vs Age")
            ax.set_xlabel("Age")
            ax.set_ylabel("Medical Charge")
            st.pyplot(fig)

        elif feature_pair == "Is there a difference in average charges between smokers and non-smokers?":
            fig, ax = plt.subplots()
            sns.boxplot(x='smoker', y='charges', data=df, ax=ax)
            ax.set_title("Medical Charges: Smokers vs Non-Smokers")
            st.pyplot(fig)

        elif feature_pair == "Does BMI impact insurance charges?":
            fig, ax = plt.subplots()
            # sns.scatterplot(x='bmi', y='charges', data=df, ax=ax)
            sns.regplot(x='bmi', y='charges', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'green'})
            ax.set_title("Charges vs BMI")
            ax.set_xlabel("BMI")
            ax.set_ylabel("Charges")
            st.pyplot(fig)

        elif feature_pair == "Do men or women pay more on average?":
            gender_charges = df.groupby('sex')['charges'].mean()
            abs_diff = abs(gender_charges[1] - gender_charges[0])
            st.write(f" **Average difference in charges (Male - Female): ₹{abs_diff:.2f}**")
            fig, ax = plt.subplots()
            sns.boxplot(x='sex', y='charges', data=df, ax=ax)
            ax.set_title("Medical Charges: Male vs Female")
            st.pyplot(fig)

        elif feature_pair == "Is there a correlation between the number of children and the insurance charges?":
            corr = df['children'].corr(df['charges'])
            st.write(f" Correlation between number of children and charges: **{corr:.4f}**")
            fig, ax = plt.subplots()
            sns.stripplot(x='children', y='charges', data=df, jitter=0.25, palette='coolwarm', ax=ax)
            ax.set_title("Distribution of Charges by Children Count")
            st.pyplot(fig)

    elif analysis_type == "Multivariate Analysis":
        option = st.selectbox("Choose multivariate scenario", [
            "How does smoking status combined with age affect medical charges?",
            "What is the impact of gender and region on charges for smokers?",
            "How do age, BMI, and smoking status together affect insurance cost?",
            "Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?"])

        if option == "How does smoking status combined with age affect medical charges?":
            fig, ax = plt.subplots()
            sns.scatterplot(x='age',y='charges',hue='smoker',data=df,palette='Set1',alpha=0.6)
            ax.set_title("Charges vs Age by Smoking Status")
            ax.set_xlabel("Age")
            ax.set_ylabel("Medical Charges")
            ax.legend(title="Smoker (1=Yes, 0=No)")
            st.pyplot(fig)

        elif option == "What is the impact of gender and region on charges for smokers?":
            smokers_df = df[df['smoker'] == 1]
            st.write("### Average Charges for Smokers by Region and Gender")
            grouped_avg = smokers_df.groupby(['region', 'sex'])['charges'].mean().unstack()
            st.dataframe(grouped_avg.style.format("₹{:.2f}"))
            fig, ax = plt.subplots(figsize=(10,5))
            sns.boxplot(x='region', y='charges', hue='sex', data=df[df['smoker']=='yes'], ax=ax)
            ax.set_title("Gender & Region vs Charges (Smokers only)")
            ax.set_xlabel("Region")
            ax.set_ylabel("Medical Charges")
            ax.legend(title="Smoker (1=Yes, 0=No)")
            st.pyplot(fig)

        elif option == "How do age, BMI, and smoking status together affect insurance cost?":
            fig, ax = plt.subplots()
            sns.scatterplot(x='age', y='bmi', hue='smoker', size='charges', sizes=(20, 300), data=df, ax=ax)
            ax.set_title("Age vs BMI Colored by Smoker and Sized by Charges")
            st.pyplot(fig)

        elif option == "Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?":
            # Step 1: Add obesity flag
            df['obese'] = (df['bmi'] > 30).astype(int)
            
            # Step 2: Define groups
            group1 = df[(df['smoker'] == 1) & (df['obese'] == 1)]  # Obese Smokers
            group2 = df[(df['smoker'] == 0) & (df['obese'] == 0)]  # Non-obese Non-Smokers
            
            # Step 3: Calculate means and difference
            mean1 = group1['charges'].mean()
            mean2 = group2['charges'].mean()
            diff = mean1 - mean2

            # Step 4: Display metrics
            st.write(f"### Charges Comparison")
            st.write(f"**Obese Smokers Avg**: ₹{mean1:.2f}")
            st.write(f"**Non-obese Non-Smokers Avg**: ₹{mean2:.2f}")
            st.write(f"**Difference**: ₹{diff:.2f}")

            # Step 5: Create group labels for plotting
            df['group'] = 'Other'
            df.loc[(df['smoker'] == 1) & (df['obese'] == 1), 'group'] = 'Obese Smoker'
            df.loc[(df['smoker'] == 0) & (df['obese'] == 0), 'group'] = 'Non-Obese Non-Smoker'
            # df['obese_smoker'] = ((df['bmi'] > 30) & (df['smoker'] == 'yes')).astype(int)
            fig, ax = plt.subplots()
            sns.boxplot(x='group',y='charges',hue='group',data=df[df['group'] != 'Other'],palette='Set3')
            ax.set_title("Charges: Obese Smokers vs Non-Obese Non-Smokers")
            st.pyplot(fig)

    elif analysis_type == "Outlier Detection":
        option = st.selectbox("Choose outlier detection scenario", [
            "Are there outliers in the charges column?",
            "Who are the individuals paying the highest costs?",
            "Are there extreme BMI values that could skew predictions?"])

        if option == "Are there outliers in the charges column?":
            st.subheader("Outlier Detection for Charges")
            Q1 = df['charges'].quantile(0.25)
            Q3 = df['charges'].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            outliers = df[df['charges'] > upper]
            st.write(f"Number of outliers in charges: {len(outliers)}")

            fig, ax = plt.subplots()
            sns.boxplot(x=df['charges'], ax=ax, color='lightgreen')
            ax.set_title("Boxplot of Charges (Outliers)")
            ax.set_xlabel("Charges")
            st.pyplot(fig)
        
        elif option == "Who are the individuals paying the highest costs?":
            st.subheader("Top 5 Most Expensive Cases")
            # Top 5 highest charges
            top5 = df.sort_values(by='charges', ascending=False).head(5)
            # Display selected columns
            st.write("### Top 5 Highest Medical Charges")
            st.dataframe(top5[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']])
            
        elif option == "Are there extreme BMI values that could skew predictions?":
            st.subheader("Extreme BMI Values")
            # Calculate IQR and bounds for BMI
            Q1_bmi = df['bmi'].quantile(0.25)
            Q3_bmi = df['bmi'].quantile(0.75)
            IQR_bmi = Q3_bmi - Q1_bmi
            lower_bmi = Q1_bmi - 1.5 * IQR_bmi
            upper_bmi = Q3_bmi + 1.5 * IQR_bmi

            # Identify BMI outliers
            bmi_outliers = df[df['bmi'] > upper_bmi]
            st.write(f" **Extreme BMI outliers**: {len(bmi_outliers)} individuals")
            fig, ax = plt.subplots()
            sns.boxplot(x=df['bmi'], color='skyblue')
            ax.set_title("Boxplot of BMI")
            ax.set_xlabel("BMI")
            st.pyplot(fig)

    elif analysis_type == "Correlation Analysis":
        option = st.selectbox("Choose Correlation Analysis", [
            "What is the correlation between numeric features like age, BMI, number of children, and charges?",
            "Which features have the strongest correlation with the target variable (charges)?"])

        if option == "What is the correlation between numeric features like age, BMI, number of children, and charges?":
            st.subheader("Correlation with Charges")
            numeric_cols = ['age', 'bmi', 'children', 'charges']
            corr_matrix = df[numeric_cols].corr()
            st.write("### Correlation Matrix (Numeric Features)")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

        elif option == "Which features have the strongest correlation with the target variable (charges)?":
            fig, ax = plt.subplots()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

# Prediction
elif menu == "Prediction":
    st.markdown("### Predict Your Medical Insurance Cost")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 100, 30)
            sex = st.selectbox("Gender", ["male", "female"])
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

        with col2:
            children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
            smoker = st.selectbox("Smoker?", ["yes", "no"])
            region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        sex_encoded = 1 if sex == "male" else 0
        smoker_encoded = 1 if smoker == "yes" else 0
        region_encoded = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}[region]

        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex_encoded,
            "bmi": bmi,
            "children": children,
            "smoker": smoker_encoded,
            "region": region_encoded
        }])

        try:
            prediction = model.predict(input_df)
            result = prediction[0] if isinstance(prediction, (np.ndarray, list, pd.Series)) else prediction
            st.success(f" Estimated Annual Insurance Cost: ₹{result:,.2f}")

            original_input = pd.DataFrame([{
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region
            }])

            with st.expander("View Your Input Details"):
                st.write(original_input)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
