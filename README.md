# Simple Python Weather App

A command-line application that fetches and displays the current weather for any city in the world using the OpenWeatherMap API.



## 📝 Description

This script prompts the user to enter a city name. It then makes a request to the OpenWeatherMap API to retrieve real-time weather data for that location. Finally, it parses the response and displays key weather information like temperature, humidity, wind speed, and general conditions in a clean, readable format.

## ✨ Features

- *Real-time Data*: Fetches up-to-the-minute weather information.
- *Global Coverage*: Get weather for any city worldwide.
- *User-Friendly Output*: Displays data in a simple and clear format.
- *Error Handling*: Provides helpful messages for common issues like invalid city names or incorrect API keys.
- *Secure API Key Input*: Uses getpass to hide the API key as it's being typed.

## ⚙ Prerequisites

Before you run the script, you need to have Python installed on your system. You will also need to install the requests library.

- *Python 3.x*
- *requests* library

You will also need a *free API key* from OpenWeatherMap.

## 🚀 Installation and Usage

1.  *Clone the repository or save the script:*
    Save the code above as weather_app.py.

2.  *Install the required library:*
    Open your terminal or command prompt and run the following command:
    bash
    pip install requests
    

3.  *Get an API Key:*
    - Go to the [OpenWeatherMap website](https://openweathermap.org/appid) and create a free account.
    - Navigate to the 'API keys' tab on your account page.
    - A default API key will be generated for you. Copy this key.
    > *Note:* It may take a few minutes for your new API key to become active.

4.  *Run the script:*
    Execute the script from your terminal:
    bash
    python weather_app.py
    

5.  *Follow the prompts:*
    - The script will first ask for your OpenWeatherMap API key. Paste the key you copied and press Enter.
    - Next, it will ask for the name of the city you want to check. Type the city name and press Enter.
    - The current weather information for the specified city will be displayed.

## 🧑‍💻 How It Works

The script uses the requests library to send an HTTP GET request to the OpenWeatherMap API endpoint. The request includes the city name and your unique API key as query parameters. The API returns a JSON object containing the weather data, which is then parsed using Python's built-in json library to extract and display the relevant information.