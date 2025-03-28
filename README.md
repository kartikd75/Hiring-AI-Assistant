
# Hiring Assistant Chatbot

## Project Overview

The **Hiring Assistant Chatbot** is an AI-powered conversational agent designed to assist candidates in exploring job opportunities at TalentScout, a recruitment agency specializing in technology placements. The chatbot engages candidates in a natural conversation, gathers their information (e.g., name, email, experience, tech stack), and assesses their technical skills through dynamically generated questions. It also provides personalized responses based on the candidate's inputs and preferences.

## Key Features:

* **Multilingual Support:** Interacts with candidates in their preferred language using translation and language detection.
* **Sentiment Analysis:** Gauges the candidate's emotions during the conversation and adjusts responses accordingly.
* **Dynamic Question Generation:** Generates technical questions tailored to the candidate's skills and experience.
* **Personalized Responses:** Uses candidate history and preferences to provide tailored and engaging interactions.
* **Data Privacy:** Ensures candidate data is handled securely and complies with data privacy standards.

## Installation Instructions

Follow these steps to set up and run the Hiring Assistant chatbot locally.

### Prerequisites

* Python 3.8 or higher
* Streamlit
* Groq API key (for LLM integration)

### Required Python libraries

* See `requirements.txt`

### Steps

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/kartikd75/Hiring-AI-Assistant.git](https://www.google.com/search?q=https://github.com/kartikd75/Hiring-AI-Assistant.git)
    cd TalentScout-hiringAssistant-chatbot
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables:**(OPTIONAL) ..didn't used in the current code.

    * Create a `.env` file in the root directory.
    * Add your Groq API key:

        ```plaintext
        GROQ_API_KEY=your_groq_api_key_here
        ```

4.  **Run the Application:**

    ```bash
    streamlit run chatbot.py
    ```

5.  **Access the Chatbot:**

    * Open your browser and navigate to `http://localhost:8501`.

## Usage Guide

### Start the Chat:

* The chatbot will greet you and ask for your name.
* Follow the prompts to provide your information (e.g., email, phone, experience, tech stack).

### Technical Questions:

* The chatbot will generate technical questions based on your skills.
* Answer the questions to the best of your ability.

### End the Conversation:

* You can end the conversation at any time by typing "exit", "quit", "goodbye", "end" or "bye".

### Admin Dashboard:

* Use the sidebar to view the candidate database (for admin purposes).

## Technical Details

### Libraries Used

* **Streamlit:** For building the web interface.
* **Groq API:** For interacting with the LLM (Llama-3.1-8b-instant).
* **TextBlob:** For sentiment analysis.
* **Deep Translator:** For multilingual support (translation).
* **Langdetect:** For language detection.
* **Pandas:** For handling candidate data.

### Model Details

* **LLM:** Llama-3.1-8b-instant (via Groq API).
* **Prompt Design:** Carefully crafted prompts ensure natural and context-aware conversations.

### Architectural Decisions

* **Modular Design:** The code is organized into classes (GroqModel, HiringAssistant) for better maintainability.
* **Session State:** Streamlit's session state is used to manage conversation history and candidate data.
* **Simulated Database:** Candidate data is stored in memory for demonstration purposes.

### Prompt Design

The chatbot uses carefully designed prompts to handle information gathering, technical question generation, and personalized responses. Here’s how prompts are crafted:

#### Greeting and Information Gathering:

* The chatbot introduces itself and asks for the candidate's name, email, phone, experience, and tech stack.
* Example prompt:

    ```plaintext
    You are a hiring assistant for TalentScout. Ask the candidate for their name and email address.
    ```

#### Technical Question Generation:

* Based on the candidate's tech stack, the chatbot generates relevant technical questions.
* Example prompt:

    ```plaintext
    Generate 3-5 technical questions based on the candidate's tech stack: {tech_stack}.
    ```

#### Sentiment-Aware Responses:

* The chatbot adjusts its tone based on the candidate's sentiment (positive, negative, or neutral).
* Example prompt:

    ```plaintext
    The candidate's sentiment is negative. Provide an empathetic response.
    ```

#### Multilingual Support:

* The chatbot detects the candidate's language and translates responses accordingly.
* Example prompt:

    ```plaintext
    Translate the following response to French: {response}.
    ```

## Challenges & Solutions

### Challenges Faced

* **Language Detection and Translation:**
    * Initially, `googletrans` was used, but it was unstable and caused errors.
    * **Solution:** Switched to `deep-translator` and `langdetect` for reliable language detection and translation.
* **Dynamic Question Generation:**
    * Generating relevant technical questions based on the candidate's tech stack was challenging.
    * **Solution:** Used the LLM to analyze the tech stack and generate tailored questions.
* **Personalized Responses:**
    * Ensuring the chatbot provides personalized responses based on candidate history was complex.
    * **Solution:** Stored candidate data in the session state and used it to tailor responses.
* **Performance Optimization:**
    * The chatbot needed to respond quickly to maintain a natural conversation flow.
    * **Solution:** Optimized API calls and used caching where possible.

## Future Enhancements

* **Sentiment Analysis:** Improve sentiment analysis to better gauge candidate emotions.
* **Multilingual Support:** Add support for more languages and improve translation accuracy.
* **UI Enhancements:** Add custom styling and interactive elements to improve the user experience.
* **Data Persistence:** Store candidate data in a database for long-term use.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
