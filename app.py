import sys
import os

# Add the path to the directory containing the notebooks folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import from notebooks.draft_notebooks.functions
from notebooks.draft_notebooks.functions import *

def main():
    
    # Load the pre-trained Prophet model
    with open('prophet.pkl', 'rb') as file:
        model = pickle.load(file)

    st.title('Sales Prediction Tool')
    
    # Load the pre-trained Prophet model
    model_path = os.path.join(os.path.dirname(__file__), 'prophet.pkl')
    
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # File uploader
    uploaded_file = st.file_uploader("Input historical sales file (2 columns: date and sales)", type="csv")

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
        fig.update_layout(title='Sales Prediction', xaxis_title='Date', yaxis_title='Sales ($)')
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

        # st.table(prediction_table)
    
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




if __name__ == '__main__':
    main()