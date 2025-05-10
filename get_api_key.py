import os

import requests
from dotenv import load_dotenv

load_dotenv()


def get_api_key():
    """
    Fetch the API key from environment variables or a .env file.

    Returns:
        str: The API key.
    """
    api_key = os.getenv("API_KEY")
    channel_id = os.getenv("CHANNEL_ID")
    print(f"API_KEY: {api_key}")
    if api_key is None or channel_id is None:
        raise ValueError(
            "API_KEY and CHANNEL_ID must be set in the environment variables."
        )
    return api_key, channel_id


def get_data():
    api_key, channel_id = get_api_key()
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.csv"
    params = {"api_key": api_key, "results": 100000}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        with open("data.csv", "wb") as file:
            file.write(response.content)
        print("Data downloaded successfully.")
    else:
        print(f"Failed to download data: {response.status_code}")


if __name__ == "__main__":
    get_data()
