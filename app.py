import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate synthetic house price data
def generate_house_data(n_samples=100):
    np.random.seed(50)
    size = np.random.normal(1400, 500, n_samples)
    price = size * 50 + np.random.normal(0, 500, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

# Train a simple linear regression model
def train_model():
    df = generate_house_data(n_samples=100)
    X = df[['size']]
    Y = df['price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

# Main Streamlit app
def main():
    st.title('🏡 House Price Prediction
    st.title('A sample project by SS')
    
    st.write("Enter the size of your house to estimate its price.")

    model = train_model()
    df = generate_house_data()

    size = st.number_input("House size (sq ft)", min_value=500, max_value=2000, value=1500)

    if st.button("Predict price"):
        predicted_price = model.predict(np.array([[size]]))
        st.success(f'💰 Estimated price: ${predicted_price[0]:,.2f}')

        fig = px.scatter(df, x="size", y="price", title="House Price vs Size")
        fig.add_scatter(x=[size], y=[predicted_price[0]],
                        mode="markers",
                        marker=dict(color="red", size=12),
                        name="Prediction")
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
