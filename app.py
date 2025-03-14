import streamlit as st
import pandas as pd
import yfinance as yf
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



load_dotenv()

tickers_df = pd.read_csv("nasdaq_tickers.csv")  # Ensure this file is present

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=200,
    timeout=None,
    max_retries=2,
    api_key=os.getenv('GROQ_API_KEY')
)

def get_ticker(company_name):
    match = tickers_df[tickers_df['Company'].str.contains(company_name, case=False, na=False)]
    return match.iloc[0]['Symbol'] if not match.empty else None

def extract_companies(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="List only the company names mentioned in the following query, separated by commas, without any extra text: {query}."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query)
    
    return [company.strip() for company in response.split(",") if company.strip()]


def fetch_stock_news(symbol):
    news_api_key = os.getenv("NEWS_API_KEY")
    
    if not news_api_key:
        return "\n## 📰 Latest News\n- ❌ API key missing."

    url = f'https://newsapi.org/v2/everything?q={symbol}&sortBy=publishedAt&apiKey={news_api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])[:5]  
        if not articles:
            return "\n## 📰 Latest News\n- No recent news."

        return "\n## 📰 Latest News\n" + "".join(
            f"- [{article.get('title', 'No Title')}]({article.get('url', '#')})\n"
            for article in articles
        )
    except requests.RequestException as e:
        return f"\n## 📰 Latest News\n- ❌ Error: {str(e)}"

def generate_ai_suggestion(report_data):
    prompt = PromptTemplate(
        input_variables=["report_data"],
        template="Based on the following stock data: {report_data}, provide an investment suggestion."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({"report_data": report_data}) 


def generate_investment_report(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')

    if todays_data.empty:
        return f"⚠ Stock data for {symbol} is unavailable."

    info = ticker.info
    report_data = {
        "Company Name": info.get("longName", symbol),
        "Sector": info.get("sector", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Revenue": info.get("totalRevenue", "N/A"),
        "Net Income": info.get("netIncomeToCommon", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A"),
        "Beta": info.get("beta", "N/A"),
        "52-Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52-Week Low": info.get("fiftyTwoWeekLow", "N/A"),
        "50-Day Moving Avg": info.get("fiftyDayAverage", "N/A"),
        "200-Day Moving Avg": info.get("twoHundredDayAverage", "N/A")
    }
    
    ai_suggestion = generate_ai_suggestion(report_data)
    stock_news = fetch_stock_news(symbol)

    return f"""
    # {report_data['Company Name']} ({symbol})
    
    ## 🏢 Company Overview
    - **Sector:** {report_data['Sector']}
    - **Market Cap:** {report_data['Market Cap']:,}
    
    ## 📈 Financial Overview
    - **Price-to-Earnings Ratio (P/E):** {report_data['P/E Ratio']}
    - **Earnings Per Share (EPS):** {report_data['EPS']}
    - **Revenue:** {report_data['Revenue']:,}
    - **Net Income:** {report_data['Net Income']:,}
    - **Dividend Yield:** {report_data['Dividend Yield']}
    
    ## 📊 Technical Indicators
    - **52-Week High:** ${report_data['52-Week High']}
    - **52-Week Low:** ${report_data['52-Week Low']}
    - **50-Day Moving Average:** {report_data['50-Day Moving Avg']}
    - **200-Day Moving Average:** {report_data['200-Day Moving Avg']}
    
    ## ⚠ Risk Assessment
    - **Beta:** {report_data['Beta']}
    
    ## 🏦 Investment Suggestion
    {ai_suggestion}

    
    {stock_news}
    """


hekl = " cjcj:"
def compare_stocks(symbol1, symbol2):
    report1 = generate_investment_report(symbol1)
    report2 = generate_investment_report(symbol2)
    
    prompt = PromptTemplate(
        input_variables=["report1", "report2"],
        template="Compare these two stock reports side by side:\n{report1}\n\n{report2}\n\nProvide a comparative investment suggestion."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    comparison_analysis = chain.run({"report1": report1, "report2": report2})


    # 📊 Stock Comparison: {symbol1} vs {symbol2}
    ## {symbol2} Report


    return f"""    
    {report1.split('Latest')[-1][0:10].replace('[T', '..').replace('News','         ')}
    # {report1} ++

    {report1.split('Latest')[-1][0:10].replace('[T', '..').replace('News','                                                 ')}
    {report2} ++
    
    ## 📊 AI-Powered Comparison
    {comparison_analysis}
    """

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.sidebar.image('wp9587602.jpg')

st.title("📈 Stock Investment Analysis Tool")
st.write('Get AI-powered investment insights for any stock!')
st.markdown('### 💵 Investment Report')
st.sidebar.header("🔍 Stock Search")


query = st.sidebar.text_input('Enter stock query:', value="Compare Apple and Tesla stocks")

if st.sidebar.button('Submit'):
    if query:
        companies = extract_companies(query)
    
        if len(companies) == 2:
            symbol1, symbol2 = get_ticker(companies[0]), get_ticker(companies[1])
            if symbol1 and symbol2:
                response = compare_stocks(symbol1, symbol2)
            else:
                response = f"❌ Could not find tickers for {companies}."
    
            c1, c2 = st.columns(2)
            c1.markdown(response.split('++')[0])
            c2.title('')
            c2.markdown(response.split('++')[1])
            st.markdown(response.split('++')[-1])
    
        elif len(companies) == 1:
            symbol = get_ticker(companies[0])
            if symbol:
                response = generate_investment_report(symbol)
            else:
                response = f"❌ Could not find ticker for {companies[0]}."
            st.markdown(response.split('Latest')[-1][0:10].replace('[p', '||').replace('News','         ')  + response)
            
        else:
            response = "❌ Could not identify valid company names in the query."
            st.markdown(response)
    
    
