import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import finnhub as fh
import datetime
from datetime import date, timedelta


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use a pipeline as a high-level helper


#initialize Finnhub API

finnhub_client = fh.Client(api_key="cohiiipr01qkmfrc5rc0cohiiipr01qkmfrc5rcg")

# Function to fetch stock profile

# Function to fetch stock company name
def fetch_stock_name(symbol):
    stock_profile = finnhub_client.company_profile2(symbol=symbol)
    stock_name = stock_profile['name']

    return stock_name

# Function to fetch stock exchange
def fetch_stock_exchange(symbol):
    stock_profile = finnhub_client.company_profile2(symbol=symbol)
    stock_exchange = stock_profile['exchange']

    return stock_exchange

# Function to fetch stock ticker
def fetch_stock_ticker(symbol):
    stock_profile = finnhub_client.company_profile2(symbol=symbol)
    stock_ticker = stock_profile['ticker']

    return stock_ticker

# Function to fetch stock country
def fetch_stock_country(symbol):
    stock_profile = finnhub_client.company_profile2(symbol=symbol)
    stock_country = stock_profile['country']

    return stock_country

# Function to fetch stock image
def fetch_stock_image(symbol):
    stock_profile = finnhub_client.company_profile2(symbol=symbol)
    stock_image = stock_profile['logo']

    return stock_image

def fetch_stock_news(symbol):
    # news for last 1 month
    today = date.today()
    day_of_week = today.weekday()  # 0 for Monday, 6 for Sunday

    # Calculate the starting date of the last week
    last_week_start = today - timedelta(days=1)
    start_date = last_week_start.strftime('%Y-%m-%d')

    # Generate a list of dates for the entire week
    dates = [last_week_start + timedelta(days=i) for i in range(7)]
    
    end_date = today.strftime('%Y-%m-%d')
    news = finnhub_client.company_news(symbol=symbol, _from=start_date, to=end_date)
    
    
    return news[:10]
    
# Function to fetch news sentiment using BERT model


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def fetch_news_sentiment(text):
    # Load the model and tokenizer
    # Load model directly
# Load model directly
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("nickmuchi/sec-bert-finetuned-finance-classification")
    model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/sec-bert-finetuned-finance-classification")

    # Create a pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # Make a prediction
    result = nlp(text)

    return result
    
        






# Function to fetch historical stock price data
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Function to visualize line chart
def visualize_line_chart(stock_data):
    fig = go.Figure(data=go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Stock Price Line Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to visualize candlestick chart
def visualize_candlestick_chart(stock_data):    
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'])])
    fig.update_layout(title='Stock Price Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to visualize bar chart
def visualize_bar_chart(stock_data):
    fig = go.Figure(data=[go.Bar(x=stock_data.index, y=stock_data['Volume'])])
    fig.update_layout(title='Stock Volume Bar Chart', xaxis_title='Date', yaxis_title='Volume')
    return fig


# Function to visualize stock prices as a triangle chart
def visualize_triangle_chart(stock_data):
    fig = go.Figure(data=go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', fill='toself'))
    fig.update_layout(title='Stock Price Triangle Chart', xaxis_title='Date', yaxis_title='Price')
    return fig

def return_sentment_emoji(sentiment):
    if sentiment == "bullish":
        return ":chart_with_upwards_trend:"
    if  sentiment == "bearish":
        return ":chart_with_downwards_trend:"
    else:
        return ":neutral_face:"



def app():
    
    # side bar
    symbol = st.sidebar.text_input("Enter Stock Symbol", value='AAPL')
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime('2021-01-01'))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime('2021-12-31'))
    chart_types = st.sidebar.multiselect("Select Chart Types", ['Line Chart', 'Candlestick Chart', 'Bar Chart', 'Triangle Chart'])
    
    stock_data = fetch_stock_data(symbol, start_date, end_date)
    
     # Visualize stock prices
    if not stock_data.empty:
        st.header(f"Stock Price Data for {symbol}")
        st.write(stock_data)

        if 'Line Chart' in chart_types:
            st.header("Stock Price Line Chart")
            fig_line_chart = visualize_line_chart(stock_data)
            st.plotly_chart(fig_line_chart)

        if 'Candlestick Chart' in chart_types:
            st.header("Stock Price Candlestick Chart")
            fig_candlestick_chart = visualize_candlestick_chart(stock_data)
            st.plotly_chart(fig_candlestick_chart)

        if 'Bar Chart' in chart_types:
            st.header("Stock Price Bar Chart")
            fig_bar_chart = visualize_bar_chart(stock_data)
            st.plotly_chart(fig_bar_chart)

        if 'Triangle Chart' in chart_types:
            st.header("Stock Price Triangle Chart")
            fig_triangle_chart = visualize_triangle_chart(stock_data)
            st.plotly_chart(fig_triangle_chart)
    
    st.divider()
            
    st.header("Company Profile")
    st.write("Company Name: ", fetch_stock_name(symbol))
    st.write("Stock Exchange: ", fetch_stock_exchange(symbol))
    st.write("Stock Ticker: ", fetch_stock_ticker(symbol))
    st.write("Stock Country: ", fetch_stock_country(symbol))
    st.write("Stock Image: ")
    st.image(fetch_stock_image(symbol),width=500 , caption='Company Logo')
    
    st.divider()
    
    st.title("Stock News")
    news = fetch_stock_news(symbol)
    for i in news:
        st.subheader(i['headline'])
        st.write(i['summary'])
        st.write(i['url'])
        st.write(i['source'])
        
    


        st.subheader(f"**_Sentiment: {fetch_news_sentiment(i['summary'])[0]['label']}{return_sentment_emoji(fetch_news_sentiment(i['summary'])[0]['label'])}_**")
        st.write('-----------------------------------')

    # st.write(fetch_stockholders(symbol))
    
    # # Fetch and display largest stockholders
    # stockholders_df = fetch_stockholders(symbol)
    # st.header("Largest Stockholders")
    # st.write(stockholders_df)

    


if __name__ == '__main__':
    app()                
    