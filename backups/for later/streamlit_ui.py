import streamlit as st
import socket

st.title("Computer Configurator")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def send_to_expert_system(command):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("localhost", 9999))
    client.send(command.encode('utf-8'))
    response = client.recv(4096).decode('utf-8')
    client.close()
    return response


def main_menu():
    st.chat_message("assistant").markdown(
        "\nWelcome to the Computer Configurator!")
    choice = st.selectbox("Please select an option:", [
                          "Configure", "View components", "Exit"])

    if choice == 'Configure':
        configure_menu()
    elif choice == 'View components':
        view_components_menu()
    elif choice == 'Exit':
        st.chat_message("assistant").markdown(
            "Exiting the application. Goodbye!")
        st.session_state.messages.append(
            {"role": "assistant", "content": "Exiting the application. Goodbye!"})


def configure_menu():
    st.chat_message("assistant").markdown(
        "\nConfiguration Menu\nType 'back' to return to the main menu.")
    task = st.text_input("Enter your task (e.g., ml, graphic design):")
    budget = st.text_input("Enter your budget (in USD):")

    if st.button("Submit"):
        st.chat_message("user").markdown(f"Task: {task}\nBudget: {budget}")
        st.session_state.messages.append(
            {"role": "user", "content": f"Task: {task}\nBudget: {budget}"})

        command = f"CONFIGURE {task} {budget}"
        output = send_to_expert_system(command)
        st.chat_message("assistant").markdown(output)
        st.session_state.messages.append(
            {"role": "assistant", "content": output})


def view_components_menu():
    st.chat_message("assistant").markdown(
        "--------------------------------\nView Components Menu\n--------------------------------")
    command = "VIEW_COMPONENTS"
    output = send_to_expert_system(command)
    st.chat_message("assistant").markdown(output)
    st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    main_menu()
