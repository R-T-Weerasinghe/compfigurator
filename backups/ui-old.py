import streamlit as st

# Set up the Streamlit UI
st.title("Simple Chat Application")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ["Bot: Hi there!"]

user_input = True

# Chat logic
if user_input:
    user_input = st.text_input("You: ", key="input", value="")
    st.session_state.chat_history.append(f"You: {user_input}")

    if user_input.lower() == "bye":
        st.session_state.chat_history.append("Bot: Goodbye!")
    else:
        st.session_state.chat_history.append(f"Bot: You said '{user_input}'")
    # st.text_input("You: ", key="input", value="")  # Clear input

# Display chat history
for chat in st.session_state.chat_history:
    st.write(chat)

