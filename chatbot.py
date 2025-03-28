import streamlit as st
import re
from groq import Groq
from datetime import datetime
import pandas as pd
from textblob import TextBlob  # For sentiment analysis
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator  # For translation

# Initialize Groq client
client = Groq(api_key="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")  # Replace with your actual API key

# Simulated Database for Candidate Data (In-memory storage)
if "candidate_database" not in st.session_state:
    st.session_state.candidate_database = []

# Model class for LLM integration using Groq
class GroqModel:
    def __init__(self):
        self.client = client
    
    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the given messages using Groq API.
        """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages
        )
        return response.choices[0].message.content

# Hiring Assistant class
class HiringAssistant:
    def __init__(self):
        self.model = GroqModel()
        self.exit_keywords = ["exit", "quit", "bye", "goodbye", "end"]
    
    def get_greeting(self):
        greeting = """
        Hello and welcome to TalentScout! My name is Emily, and I'm your hiring assistant here at TalentScout. 
        It's a pleasure to connect with you today.  

        Before we dive in, I'd like to confirm that you're interested in exploring exciting technology job opportunities 
        with our clients. My goal is to help you find the perfect role that aligns with your skills, experience, 
        and career aspirations.  

        To get started, could you please tell me your full name? This will help me personalize our conversation 
        and ensure I address you correctly.  

        Once I have your name, I'll ask a few questions about your background, expertise, and what you're looking 
        for in your next career move. Let's work together to find the best opportunities for you!
        """
        return greeting
    
    def get_farewell(self):
        prompt = """
        You are a hiring assistant for TalentScout.
        The candidate is ending the conversation.
        Provide a warm, professional farewell message.
        Thank them for their time and mention that the TalentScout team will be in touch if there's a suitable match.
        """
        return self.model.generate_response([{"role": "user", "content": prompt}])
    
    def is_exit_keyword(self, text):
        return any(keyword in text.lower() for keyword in self.exit_keywords)
    
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the candidate's input using TextBlob.
        Returns a sentiment score between -1 (negative) and 1 (positive).
        """
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    
    # Ensure consistent language detection
    DetectorFactory.seed = 0  # Add this line to make langdetect deterministic

    def detect_language(self, text):
        """
        Detect the language of the candidate's input using langdetect.
        Default to English if detection fails or is ambiguous.
        """
        try:
            # Only detect language if the input is long enough
            if len(text.strip().split()) >= 3:  # Require at least 3 words for reliable detection
                detected_lang = detect(text)
                return detected_lang
            else:
                return "en"  # Default to English for short inputs
        except Exception as e:
            print(f"Error detecting language: {e}")
            return "en"  # Default to English if detection fails
    
    def translate_text(self, text, src_lang, dest_lang="en"):
        """
        Translate text from source language to destination language using Google Translate.
        """
        try:
            translated = GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
            return translated
        except Exception as e:
            print(f"Error translating text: {e}")
            return text  # Return original text if translation fails
    
    def process_input(self, user_input, current_state, candidate_info, asked_questions):
        # Detect the language of the user's input
        src_lang = self.detect_language(user_input)
    
        # Translate the input to English for processing (if not already in English)
        if src_lang != "en":
            user_input_en = self.translate_text(user_input, src_lang, "en")
        else:
            user_input_en = user_input
        
        # Analyze sentiment of the translated input
        sentiment_score = self.analyze_sentiment(user_input_en)
        
        # Generate a response based on the current state
        messages = [
            {"role": "system", "content": f"""
            You are a hiring assistant for TalentScout. The current state of the conversation is: {current_state}. 
            The candidate has provided the following information so far: {candidate_info}. 
            Respond appropriately based on the user input: {user_input_en}.
            
            Additionally, the sentiment analysis of the candidate's input is: {"positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"}.
            Adjust your tone accordingly to be empathetic and supportive if the sentiment is negative, or enthusiastic if the sentiment is positive.
            """},
            {"role": "user", "content": user_input_en}
        ]
        response_en = self.model.generate_response(messages)
        
        # Translate the response back to the candidate's language (if not English)
        if src_lang != "en":
            response = self.translate_text(response_en, "en", src_lang)
        else:
            response = response_en
        
        # Rest of the logic remains the same
        if current_state == "greeting":
            return self.collect_name(user_input, candidate_info, response)
        elif current_state == "collect_name":
            return self.collect_email(user_input, candidate_info, response)
        elif current_state == "collect_email":
            return self.collect_phone(user_input, candidate_info, response)
        elif current_state == "collect_phone":
            return self.collect_experience(user_input, candidate_info, response)
        elif current_state == "collect_experience":
            return self.collect_position(user_input, candidate_info, response)
        elif current_state == "collect_position":
            return self.collect_location(user_input, candidate_info, response)
        elif current_state == "collect_location":
            return self.collect_tech_stack(user_input, candidate_info, response)
        elif current_state == "collect_tech_stack":
            return self.generate_tech_questions(user_input, candidate_info, response)
        elif current_state == "tech_questions":
            return self.process_tech_answers(user_input, candidate_info, asked_questions, response)
        elif current_state == "wrap_up":
            return self.end_conversation(user_input, candidate_info, response)
        else:
            return {
                "message": "I'm not sure how to proceed. Let's start over. What's your name?",
                "new_state": "collect_name",
                "candidate_info": candidate_info
            }
    
    def collect_name(self, user_input, candidate_info, response):
        name = self._extract_name(user_input)
        if name:
            candidate_info["name"] = name
            return {
                "message": response,
                "new_state": "collect_email",
                "candidate_info": candidate_info
            }
        else:
            return {
                "message": "I didn't catch your name. Could you please tell me your full name?",
                "new_state": "collect_name",
                "candidate_info": candidate_info
            }
    
    def collect_email(self, user_input, candidate_info, response):
        email = self._extract_email(user_input)
        if email:
            candidate_info["email"] = email
            return {
                "message": response,
                "new_state": "collect_phone",
                "candidate_info": candidate_info
            }
        else:
            # Use LLM to generate a polite response for invalid email
            prompt = f"""
            The candidate provided an invalid email: {user_input}.
            Politely ask them to provide a valid email address.
            """
            llm_response = self.model.generate_response([{"role": "user", "content": prompt}])
            return {
                "message": llm_response,
                "new_state": "collect_email",
                "candidate_info": candidate_info
            }
    
    def collect_phone(self, user_input, candidate_info, response):
        phone = self._extract_phone(user_input)
        if phone:
            candidate_info["phone"] = phone
            return {
                "message": response,
                "new_state": "collect_experience",
                "candidate_info": candidate_info
            }
        else:
            # Use LLM to generate a polite response for invalid phone
            prompt = f"""
            The candidate provided an invalid phone number: {user_input}.
            Politely ask them to provide a valid phone number.
            """
            llm_response = self.model.generate_response([{"role": "user", "content": prompt}])
            return {
                "message": llm_response,
                "new_state": "collect_phone",
                "candidate_info": candidate_info
            }
    
    def collect_experience(self, user_input, candidate_info, response):
        experience = self._extract_experience(user_input)
        if experience is not None:
            candidate_info["experience"] = experience
            return {
                "message": response,
                "new_state": "collect_position",
                "candidate_info": candidate_info
            }
        else:
            # Use LLM to generate a polite response for invalid experience
            prompt = f"""
            The candidate provided an invalid number of years of experience: {user_input}.
            Politely ask them to provide a valid number of years.
            """
            llm_response = self.model.generate_response([{"role": "user", "content": prompt}])
            return {
                "message": llm_response,
                "new_state": "collect_experience",
                "candidate_info": candidate_info
            }
    
    def collect_position(self, user_input, candidate_info, response):
        candidate_info["position"] = user_input.strip()
        return {
            "message": response,
            "new_state": "collect_location",
            "candidate_info": candidate_info
        }
    
    def collect_location(self, user_input, candidate_info, response):
        candidate_info["location"] = user_input.strip()
        return {
            "message": response,
            "new_state": "collect_tech_stack",
            "candidate_info": candidate_info
        }
    
    def collect_tech_stack(self, user_input, candidate_info, response):
        candidate_info["tech_stack"] = user_input.strip()
        return {
            "message": response,
            "new_state": "tech_questions",
            "candidate_info": candidate_info,
            "asked_questions": []
        }
    
    def generate_tech_questions(self, tech_stack, candidate_info, response):
        messages = [
            {"role": "system", "content": f"Generate 3-5 technical questions based on the candidate's tech stack: {tech_stack}. Make the questions relevant, varied in difficulty, and focused on practical application."},
            {"role": "user", "content": "Generate technical questions."}
        ]
        questions = self.model.generate_response(messages)
        return {
            "message": f"Here are some technical questions based on your skills:\n\n{questions}\n\nPlease answer these questions to the best of your ability.",
            "new_state": "tech_questions",
            "candidate_info": candidate_info,
            "asked_questions": questions.split("\n")
        }
    
    def process_tech_answers(self, user_input, candidate_info, asked_questions, response):
        if not hasattr(candidate_info, "tech_answers"):
            candidate_info["tech_answers"] = [user_input]
        else:
            candidate_info["tech_answers"].append(user_input)
        
        # Save candidate data to the simulated database
        candidate_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.candidate_database.append(candidate_info)
        
        return {
            "message": response,
            "new_state": "wrap_up",
            "candidate_info": candidate_info
        }
    
    def end_conversation(self, user_input, candidate_info, response):
        return {
            "message": response,
            "new_state": "end",
            "candidate_info": candidate_info
        }
    
    # Helper methods for extracting information
    def _extract_name(self, text):
        words = text.strip().split()
        if len(words) >= 1:
            return " ".join(words)
        return None
    
    def _extract_email(self, text):
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None
    
    def _extract_phone(self, text):
        pattern = r"(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{10})"
        match = re.search(pattern, text)
        if match:
            return match.group()
        return None
    
    def _extract_experience(self, text):
        pattern = r"(\d+)"
        match = re.search(pattern, text)
        if match:
            return int(match.group())
        return None

# Main Streamlit application
def main():
    st.set_page_config(page_title="TalentScout Hiring Assistant", page_icon="🤖", layout="wide")
    
    st.title("TalentScout Hiring Assistant")
    st.markdown("#### Tech Recruitment Initial Screening")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.candidate_info = {
            "name": None,
            "email": None,
            "phone": None,
            "experience": None,
            "position": None,
            "location": None,
            "tech_stack": None
        }
        st.session_state.current_state = "greeting"
        st.session_state.asked_questions = []
    
    # Initialize the assistant
    assistant = HiringAssistant()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initial greeting
    if not st.session_state.messages:
        greeting = assistant.get_greeting()
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        with st.chat_message("assistant"):
            st.markdown(greeting)
    
    # Get user input
    user_input = st.chat_input("Type your response here...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Check for exit keywords
        if assistant.is_exit_keyword(user_input):
            farewell = assistant.get_farewell()
            st.session_state.messages.append({"role": "assistant", "content": farewell})
            with st.chat_message("assistant"):
                st.markdown(farewell)
            return
        
        # Process user input based on current state
        response = assistant.process_input(user_input, st.session_state.current_state, st.session_state.candidate_info, st.session_state.asked_questions)
        
        # Update session state
        st.session_state.current_state = response["new_state"]
        st.session_state.candidate_info = response["candidate_info"]
        if "asked_questions" in response:
            st.session_state.asked_questions = response["asked_questions"]
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["message"]})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response["message"])
        
        # Auto-scroll to bottom
        st.rerun()

    # Display candidate database (for admin purposes)
    if st.sidebar.checkbox("View Candidate Database (Admin)"):
        st.sidebar.write("### Candidate Database")
        if st.session_state.candidate_database:
            df = pd.DataFrame(st.session_state.candidate_database)
            st.sidebar.dataframe(df)
        else:
            st.sidebar.write("No candidate data available yet.")

if __name__ == "__main__":
    main()
