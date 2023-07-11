## Introduction 📖

The language model-driven project utilizes the LangChain framework, an in-memory database, and Streamlit for serving the app.

This innovative project harnesses the power of LangChain, a transformative framework for developing applications powered by language models. With LangChain at its core, the application offers a chat interface that communicates with text files, leveraging the capabilities of OpenAI's language models.

In this project, the language model seamlessly connects to other data sources, enabling interaction with its environment and aligning with the principles of the LangChain framework. Users can upload text files, which are then processed and stored in the in-memory database. These files serve as the basis for engaging in meaningful conversations through the AI-powered chat interface.

The project relies on Streamlit, a powerful tool for serving web applications. Streamlit provides a seamless and efficient way to showcase the functionality of the application.

To learn more about the technologies used in this project:

- [LangChain](https://langchain.org): Check out LangChain's website for detailed information.
- [Streamlit](https://streamlit.io): Explore Streamlit's website to learn more about its capabilities for serving web applications.


## Prerequisites ✅

Make sure you have an OpenAI API key 🗝️
- You can obtain an API key by signing up for OpenAI at [https://openai.com](https://openai.com)

Make a copy of `.env.example` file and put your OpenAI API key in the appropriate field

Make sure you have `python3.11` installed in your system 🐍
- You can download and install Python 3.11 from the official Python website: [https://www.python.org/downloads](https://www.python.org/downloads)

## Setup 💻
Run the following command to install the `virtualenv` package:
```
python3.11 -m pip install virtualenv
```

Create a virtual environment using the following command:
```
python3.11 -m virtualenv .venv
```

Activate the virtual environment using the following command:
```
source .venv/bin/activate
```

Install `pipenv` using:
```
pip install pipenv
```

Install packages from `Pipfile` using:
```
pipenv install
```

## Running the app 🌐

Once the packages are installed, run the following command to start the application:
```
streamlit run app.py
```

It will open a webpage in your browser 🌐
- If it doesn't automatically open, you can manually visit `http://localhost:8501` in your browser.

Browse and select a `.csv` file with the source information, and enter any query regarding the source provided.

Click on the submit button to generate and see a response for your query. 👍

Make sure to properly configure your `.env` file with the API key and other necessary environment variables before running the application.

## Demo link 🔗
https://langchain-chat-with-csv-files.streamlit.app/

## Also check these out 👀
- [Langchain chat with TXT file](https://github.com/Rohan-Jalil/langchain-chat-with-txt-files)
- [Langchain chat with PDF file](https://github.com/Rohan-Jalil/langchain-chat-with-pdf-files)

## Credits 🙌

1. [LangChain](https://langchain.org): LangChain is a transformative framework that empowers the language model capabilities, allowing for the development of applications driven by language models.
2. [OpenAI](https://openai.com): OpenAI provides state-of-the-art language models that power the chat interface, enabling natural and meaningful conversations with text files.
3. [Streamlit](https://streamlit.io): Streamlit is a flexible and user-friendly framework for serving web applications, making it easy to showcase and interact with the features of the application.