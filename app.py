import sys
import os


# Now import from notebooks.draft_notebooks.functions
from functions import *

def main():
    st.title('Olist Prediction App')

    tab1, tab2 = st.tabs(["Sales Prediction", "Estimated Delivery Date Prediction"])

    with tab1:
        sales_prediction()

    with tab2:
        delivery_date_prediction()


def sales_prediction():
    st.header('Sales Prediction')
    st.divider() 
    
    # Load the pre-trained Prophet model
    model_path = os.path.join(os.path.dirname(__file__), 'prophet.pkl')
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # File uploader
    uploaded_file = st.file_uploader("Please input a historical sales file (2 columns: date and sales)", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        data['date'] = pd.to_datetime(data['date'])
        data = data.rename(columns={'date': 'ds', 'sales': 'y'})

        # Prediction period slider
        prediction_period = st.slider("Prediction period", min_value=5, max_value=365, value=30)

        # Make prediction
        future_dates = model.make_future_dataframe(periods=prediction_period)
        forecast = model.predict(future_dates)

        # Create and display the chart
        fig = plot_plotly(model, forecast)
        fig.update_layout(title='Sales Prediction', xaxis_title='Date', yaxis_title='Sales (R$)')
        st.plotly_chart(fig)

        # Display the prediction table
        st.subheader('Prediction Table')
        prediction_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_period)
        
        # Format the date and round the numerical values
        prediction_table['ds'] = prediction_table['ds'].dt.date
        prediction_table['yhat'] = prediction_table['yhat'].round(2)
        prediction_table['yhat_lower'] = prediction_table['yhat_lower'].round(2)
        prediction_table['yhat_upper'] = prediction_table['yhat_upper'].round(2)
        
        prediction_table = prediction_table.rename(columns={
            'ds': 'Date',
            'yhat': 'Predicted Sales',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        })

        # Display the table with a scrollbar
        st.dataframe(prediction_table, height=400)  

        # Download button for the prediction table
        csv = prediction_table.to_csv(index=False)
        st.download_button(
            label="Download Prediction Table as CSV",
            data=csv,
            file_name="prediction_table.csv",
            mime="text/csv",
        )    
    else:
        st.write("Please upload a CSV file to start the prediction.")




def delivery_date_prediction():
    st.header('Estimated Delivery Date Prediction')
    st.divider() 

    # Load necessary data
    customers_df = pd.read_csv('data/customers.csv')
    sellers_df = pd.read_csv('data/sellers.csv')
    geolocation_df = pd.read_csv('data/geolocation.csv')

    # Load the pre-trained model
    with open('eta_prediction.pkl', 'rb') as file:
        model = pickle.load(file)

    # Input fields
    st.write("Please enter the purchase date in the format DD/MM/YYYY")
    purchase_date = st.date_input("Purchase date", format="DD/MM/YYYY")
    purchase_hour = st.number_input("Purchase hour (0-23)", min_value=0, max_value=23, value=12)
    customer_id = st.text_input("Customer ID")
    seller_id = st.text_input("Seller ID")

    if st.button("Predict Delivery Date"):
        # Process inputs
        reference_date = datetime(2016, 9, 15)
        days_since_reference = (purchase_date - reference_date.date()).days

        # Get customer and seller information
        try:
            customer_info = customers_df[customers_df['customer_id'].astype(str) == str(customer_id)].iloc[0]
            customer_unique_index = customer_info['unique_index']
            
            seller_info = sellers_df[sellers_df['seller_id'].astype(str) == str(seller_id)].iloc[0]
            seller_unique_index = seller_info['unique_index']
        except IndexError:
            st.error("Customer ID or Seller ID not found. Please check the IDs.")
            return

        # Get geolocation information and calculate distance
        try:
            customer_location = geolocation_df[geolocation_df['unique_index'] == customer_unique_index].iloc[0]
            seller_location = geolocation_df[geolocation_df['unique_index'] == seller_unique_index].iloc[0]
            
            distance = euclidian_distance(customer_location['latitude'], seller_location['latitude'], 
                                          customer_location['longitude'], seller_location['longitude'])
        except IndexError:
            st.error("Geolocation information not found. Please check the customer and seller IDs.")
            return

        # Create input dataframe with the specified order of columns
        input_data = pd.DataFrame({
            'purchase_date': [days_since_reference],
            'purchase_hour': [purchase_hour],
            'distance': [distance]
        })

        # Apply cyclical encoding
        input_data = cyclical_encoding(input_data, 'purchase_hour', 24)
        input_data = cyclical_encoding(input_data, 'purchase_date', 365)

        # Reorder columns to match the specified order
        input_data = input_data[['purchase_date', 'purchase_hour', 'distance', 
                                 'purchase_hour_sin', 'purchase_hour_cos',
                                 'purchase_date_sin', 'purchase_date_cos']]

        # Display input data for debugging
        # st.write("Input data for prediction:")
        # st.write(input_data)

        # Make prediction
        try:
            if isinstance(model, xgb.XGBRegressor):
                prediction = model.predict(input_data)
            elif isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(input_data)
                prediction = model.predict(dmatrix)
            else:
                st.error(f"Unsupported model type: {type(model)}")
                return
            
            estimated_delivery_date = purchase_date + pd.Timedelta(days=int(prediction[0]))
            st.success(f"Estimated delivery date: {estimated_delivery_date.strftime('%d/%m/%Y')}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Model information:")
            st.write(type(model))
            st.write(model)


if __name__ == '__main__':
    main()