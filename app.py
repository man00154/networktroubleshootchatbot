import streamlit as st
import google.generativeai as genai
import os
import random
import time

# ==============================================================================
# I. APPLICATION CONFIGURATION & INITIALIZATION
# ==============================================================================

# Set the page configuration for the Streamlit app
st.set_page_config(
    page_title="MANISH - GenAI Network Troubleshooting Chatbot",
    page_icon="🤖",
    layout="wide",
)

# Set the model name from user requirements
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Retrieve the API key from Streamlit secrets
# This is the recommended way to handle secrets in Streamlit Cloud.
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error(
        "API key not found. Please add your `GEMINI_API_KEY` to the `secrets.toml` file or "
        "as a secret in your Streamlit Cloud app settings."
    )
    st.stop() # Stop the app if the key is missing

# Configure the genai library with the API key
genai.configure(api_key=API_KEY)

# ==============================================================================
# II. SIMPLE IN-MEMORY "VECTOR DATABASE" & RAG LOGIC
# ==============================================================================

# This is a mock-up of a lightweight vector database.
# In a real application, you would use an actual vector database (e.g., ChromaDB, Pinecone)
# and text embeddings to perform a semantic search.
# We'll use a simple keyword-matching approach for demonstration.
NETWORK_TROUBLESHOOTING_KB = {
    "no internet": """
    ### Troubleshooting 'No Internet Connection'

    1.  **Check Physical Connections:** Ensure all cables (Ethernet, power) are securely plugged into the modem, router, and your computer. Look for any damaged or loose cables.
    2.  **Restart Devices:**
        * Unplug the power from your modem and router.
        * Wait for 30 seconds.
        * Plug the modem back in and wait for all lights to stabilize.
        * Plug the router back in and wait for it to boot up.
        * Restart your computer or device.
    3.  **Check Indicator Lights:** On your modem and router, a solid green or blue light usually indicates a healthy connection. A red or blinking light often signals a problem. Check the manual for specific meanings.
    4.  **Test with Another Device:** Try connecting to the internet with a different device (e.g., phone or tablet) to see if the issue is with your computer or the network itself.
    5.  **Run Network Troubleshooter:** On Windows, use the built-in troubleshooter. On macOS, use Wireless Diagnostics. These can often identify common issues.
    6.  **Contact Your ISP:** If all else fails, the problem may be on your Internet Service Provider's end. Contact their support for assistance.
    """,
    "slow network": """
    ### Troubleshooting 'Slow Network Speed'

    1.  **Run a Speed Test:** Use a reliable service like speedtest.net to measure your current download and upload speeds. This confirms if the issue is with your overall connection speed.
    2.  **Reduce Connected Devices:** A large number of devices using the network simultaneously can slow it down. Disconnect unused devices.
    3.  **Check for Background Downloads:** Ensure no large files are downloading or updating in the background on any device.
    4.  **Restart Your Router:** Rebooting the router can clear its cache and improve performance.
    5.  **Check Wi-Fi Signal Strength:** Move closer to your router or use a Wi-Fi analyzer app to check for signal interference.
    6.  **Update Router Firmware:** Check your router's manufacturer website for any available firmware updates, which can improve performance and security.
    """,
    "wi-fi not working": """
    ### Troubleshooting 'Wi-Fi Connection Issues'

    1.  **Check Wi-Fi Switch:** Many laptops and some desktops have a physical switch or key combination to turn Wi-Fi on or off.
    2.  **Forget and Reconnect:** On your device, "forget" the Wi-Fi network and then try to connect to it again, entering the password as if it were a new connection.
    3.  **Verify Password:** Double-check that you are entering the correct Wi-Fi password.
    4.  **Move Closer to the Router:** Physical distance and obstacles can degrade Wi-Fi signal quality.
    5.  **Check for Interference:** Other electronic devices, like microwaves and cordless phones, can interfere with your Wi-Fi signal.
    """,
    "ip address": """
    ### Finding and Renewing Your IP Address

    **To find your IP address:**
    * **Windows:** Open Command Prompt and type `ipconfig`.
    * **macOS:** Open Terminal and type `ifconfig` or check System Settings > Network.
    * **Linux:** Open Terminal and type `ifconfig` or `ip addr show`.

    **To renew your IP address:**
    * **Windows:** Open Command Prompt and type `ipconfig /release` followed by `ipconfig /renew`.
    * **macOS / Linux:** Open Terminal and type `sudo dhclient -r` followed by `sudo dhclient`.
    """,
}

def get_relevant_info(query):
    """
    A simple RAG function that simulates retrieving relevant information
    from a knowledge base based on keywords in the user's query.
    In a real app, this would use a vector similarity search.
    """
    query = query.lower()
    for keyword, content in NETWORK_TROUBLESHOOTING_KB.items():
        if keyword in query:
            return content
    return "No specific guide found. I will try to answer based on my general knowledge."


# ==============================================================================
# III. STREAMLIT UI & CHAT LOGIC
# ==============================================================================

# Display the title of the app
st.title("GenAI Network Troubleshooting Chatbot")
st.markdown("Hello! I'm your GenAI network assistant. I can help you troubleshoot common network issues.")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.gemini_history = []
    
# Create a GenerativeModel instance
# This handles the conversation history automatically
model = genai.GenerativeModel(model_name=MODEL_NAME)
chat = model.start_chat(history=st.session_state.gemini_history)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you with your network issue?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.gemini_history.append({"role": "user", "parts": [{"text": prompt}]})

    # Simulate RAG by getting relevant info and augmenting the prompt
    retrieved_info = get_relevant_info(prompt)
    
    # Construct the final prompt for the LLM
    # This combines the retrieved info with the user's query
    augmented_prompt = (
        f"You are a helpful and expert network troubleshooting assistant. "
        f"Use the following information to guide the user and provide a detailed, step-by-step solution. "
        f"Do not invent information beyond the context or your general knowledge.\n\n"
        f"Context:\n{retrieved_info}\n\n"
        f"User's request:\n{prompt}"
    )

    # Use a loading spinner while waiting for the response
    with st.spinner("Thinking..."):
        try:
            # Send the augmented prompt to the Gemini API
            response = chat.send_message(augmented_prompt, stream=True)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Stream the response back to the user
                for chunk in response:
                    # Append each chunk to the full response
                    full_response += chunk.text
                    time.sleep(0.02)  # Simulate typing speed
                    message_placeholder.markdown(full_response + "▌")  # Add a blinking cursor
                message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.gemini_history.append({"role": "model", "parts": [{"text": full_response}]})

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.warning("Please try again or rephrase your question.")


# ==============================================================================
# IV. EXPLANATION OF CONCEPTS (FOR THE USER)
# ==============================================================================

# Add an expandable section for advanced concepts
with st.expander("Understanding the Technology"):
    st.markdown(
        """
        **Simple "Vector DB":** In a real-world application, the `NETWORK_TROUBLESHOOTING_KB` would be a
        large, external database. When you ask a question, your query would be converted into a
        vector (a list of numbers) and used to search the database for the most semantically
        similar documents. This process is called Retrieval-Augmented Generation (RAG).
        
        **Fine-Tuning:** The model used here (`gemini-2.0-flash-lite`) is a base model. "Fine-tuning"
        involves training a model on a very specific dataset (e.g., thousands of network logs and solutions)
        to make its responses even more specialized and accurate for a particular task. This is an
        offline process that would occur before deployment.
        
        **Multi-Modal AI:** The Gemini family of models is multi-modal, meaning they can process
        and understand more than just text. They can also work with images, audio, and video.
        For network troubleshooting, you could extend this app to allow a user to upload a screenshot
        of an error message or a picture of their router's lights. The model could then analyze
        the image to provide a more targeted solution. This is a potential future improvement.
        """
    )
