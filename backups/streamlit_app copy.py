import streamlit as st
from main import ComputerConfigurator, UserPreferences

def add_message(role: str, content: str):
    """Add message to chat history"""
    st.session_state.messages.append({"role": role, "content": content})

def init_session_state():
    """Initialize session state variables"""
    if 'configurator' not in st.session_state:
        st.session_state.configurator = ComputerConfigurator()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        tasks = list(st.session_state.configurator.task_requirements.keys())
        task_list = "\n".join([f"{i}. {task}" for i, task in enumerate(tasks, 1)])
        welcome_msg = f"Welcome! Please enter your desired task from the following options:\n\n{task_list}\n\nOr press Enter for a general build."
        add_message("assistant", welcome_msg)
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'task'
    if 'task' not in st.session_state:
        st.session_state.task = None
    if 'budget' not in st.session_state:
        st.session_state.budget = None
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

def handle_task_input(prompt: str):
    """Handle task input validation and state update"""
    is_valid, validated_task, suggestions = st.session_state.configurator.input_handler.validate_task(prompt)
    
    if is_valid:
        st.session_state.task = validated_task
        st.session_state.current_step = 'budget'
        add_message("assistant", f"Task set to: {validated_task}\nPlease enter your budget (or press Enter for unlimited):")
    else:
        suggestion_list = "\n".join([f"- {s}" for s in suggestions])
        add_message("assistant", f"Invalid task. Did you mean one of these?\n{suggestion_list}")

def handle_budget_input(prompt: str):
    """Handle budget input and generate configuration"""
    is_valid, validated_budget = st.session_state.configurator.input_handler.validate_budget(prompt)
    
    if is_valid:
        st.session_state.budget = validated_budget
        add_message("assistant", f"Configuring system for {st.session_state.task} with budget ${validated_budget:,.2f}")
        
        # Generate configuration
        configurator = st.session_state.configurator
        configurator.reset()
        configurator.declare(UserPreferences(task=st.session_state.task, budget=validated_budget))
        configurator.run()
        
        if configurator.best_config:
            # Display best configuration
            best_config = configurator.format_configuration(configurator.best_config, include_explanations=True)
            add_message("assistant", f"Best Configuration:\n```\n{best_config}\n```")
            
            # Display alternatives
            for i, alt in enumerate(configurator.alternative_configs[:3], 1):
                alt_config = configurator.format_configuration(alt)
                add_message("assistant", f"Alternative {i}:\n```\n{alt_config}\n```")
        else:
            add_message("assistant", "No valid configurations found for your requirements.")
        
        # Reset for next configuration
        st.session_state.current_step = 'task'
        st.session_state.task = None
        st.session_state.budget = None
    else:
        add_message("assistant", "Invalid budget. Please enter a positive number.")

def on_message_submit():
    """Handle message submission"""
    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = ""  # Clear input
        
        # Add user message
        add_message("user", prompt)
        
        # Process input based on current step
        if st.session_state.current_step == 'task':
            handle_task_input(prompt)
        elif st.session_state.current_step == 'budget':
            handle_budget_input(prompt)

def main():
    st.title("Computer Configuration Expert System")
    init_session_state()
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input with callback
    st.text_input(
        "Type your message here...",
        key="user_input",
        on_change=on_message_submit
    )
    
    # Reset button
    if st.button("Start New Configuration"):
        st.session_state.messages = []
        st.session_state.current_step = 'task'
        st.session_state.task = None
        st.session_state.budget = None
        st.rerun()

if __name__ == "__main__":
    main()