from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
from gtts import gTTS
import requests
import os
import json
import tempfile  
import pygame  
import io


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
weathermap_api_key = os.getenv("WEATHERMAP_API_KEY")

#Weather API function
def get_current_weather(latitude, longitude):
    """Get the current weather in a given latitude and longitude"""
    try:
        base = "https://api.openweathermap.org/data/2.5/weather"
        request_url = f"{base}?lat={latitude}&lon={longitude}&appid={weathermap_api_key}&units=metric"
        response = requests.get(request_url)
        
        if response.status_code == 200:
            result = {
                "latitude": latitude,
                "longitude": longitude,
                **response.json()["main"]
            }
            return json.dumps(result, indent=4)
        else:
            return f"Failed to fetch weather data. Status Code: {response.status_code}"
    except Exception as e:
        return f"Error: An exception occurred while fetching weather data. Details: {e}"
    

def text_to_speech(text):
    """Convert text to speech and play it without using pygame."""
    try:
        tts = gTTS(text=text, lang="en")
        
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
            tts.save(temp_audio_file.name)
            
            # Play the audio in Streamlit
            st.audio(temp_audio_file.name , autoplay=True)
    
    except Exception as e:
        st.error(f"Error during TTS playback: {e}")

    
#Wrap it in a LangChain Tool
weather_tool = Tool(
    name="current_weather_tool",
    func=lambda location: get_current_weather(
        *map(float, location.strip().replace("'", "").split(","))
    ),
    description=(
        "Fetches the real-time weather data for a given latitude and longitude. "
        "Provide input as 'latitude,longitude' (e.g., '33.6995,73.0363')."
        "If the format is incorrect, it will guide you politely."
    )
)

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key , model="gpt-4o-mini")
# Step 3: Initialize Memory
memory = ConversationBufferMemory(llm = llm ,max_token_limit=100 , memory_key="chat_history")

system_message = SystemMessagePromptTemplate.from_template("""
          You are WeatherBot, an intelligent assistant. You maintain context across conversations.

            1. If the user asks a general question like "Who are you?" or "How are you?", reply politely.
            2. If a user provides location details, use the `current_weather_tool` to fetch the weather.
            3. Maintain conversation context using `chat_history` to give responses based on previous user queries or weather conditions.

            Example:
            - User: What's the weather in Islamabad?
            - Bot: The temperature in Islamabad is 28°C with clear skies.
            - User: Suggest a hairstyle for this weather.
            - Bot: Since it's warm, short hairstyles like a crew cut or undercut would be great.

            Current Chat History:
            ```chat_history```                                                                                                                                                                          
""")


human_message = HumanMessagePromptTemplate.from_template("{user_query}")

#print("---------------human-message---------------",human_message)

chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

#print("---------------chat_prompt---------------",chat_prompt)

#Chat Agent Setup
tools = [weather_tool]

# Create an agent using initialize_agent
agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=False,
    handle_parsing_errors=True 
)

# Streamlit app UI
page_bg_img = """
  <style>
  [data-testid="stAppViewContainer"]{
    background-image: url("https://images.unsplash.com/photo-1500964757637-c85e8a162699?q=80&w=1806&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: local;
}

[data-testid="stHeader"] {
background: rgba(0,0,0,0);
}
  </style>
""" 
st.markdown(page_bg_img , unsafe_allow_html = True)


st.title("Weather Chatbot 🌦️")
st.write("Hi! 👋 Ask me about the weather")

if "message" not in st.session_state:
    st.session_state.message = []
if "query" not in st.session_state:
    st.session_state.query = ""

# Function to process user query
def handle_query():
    user_query = st.session_state.query.strip()
    print("-----user_query------",user_query)
    if user_query:
        st.session_state.message.append({"role": "user", "content": user_query})

        response_stream = agent_chain.run(input=user_query)
        print("---------response--------------", response_stream)
        bot_response = ""
        if hasattr(response_stream, "__iter__") and not isinstance(response_stream, str):
            for chunk in response_stream:
                if hasattr(chunk, "choices") and hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                    bot_response += chunk.choices[0].delta.content
        else:
            bot_response = response_stream 

        st.session_state.message.append({"role": "assistant", "content": bot_response})

        st.session_state.bot_response = bot_response

        st.session_state.query = ""


with st.container():
    st.write("### Conversation:")
    for chat in st.session_state.message:
        if chat["role"] == "user":
            st.markdown(f"👤 **You**: {chat['content']}")
        else:
            st.markdown(f"🤖 **Bot**: {chat['content']}")

st.text_input(
    "Your Query:",
    placeholder="E.g., What's the weather in Islamabad?",
    key="query",
    on_change=handle_query
)

if "bot_response" in st.session_state and st.session_state.bot_response:
    text_to_speech(st.session_state.bot_response)
    st.session_state.bot_response = ""  
