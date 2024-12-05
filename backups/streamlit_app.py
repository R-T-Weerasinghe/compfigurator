import streamlit as st
from main import ComputerConfigurator, UserPreferences


def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message only once at initialization
        tasks = list(st.session_state.configurator.task_requirements.keys())
        task_list = "\n".join([f"{i}. {task}" for i, task in enumerate(tasks, 1)])
        welcome_msg = f"Welcome! Please enter your desired task from the following options:\n\n{task_list}\n\nOr press Enter for a general build."
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'task'  # or 'budget'
    if 'task' not in st.session_state:
        st.session_state.task = None
    if 'budget' not in st.session_state:
        st.session_state.budget = None
    if 'configurator' not in st.session_state:
        st.session_state.configurator = ComputerConfigurator()

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

def main():
    st.title("Computer Configuration Expert System")

    # Initialize configurator first
    if 'configurator' not in st.session_state:
        st.session_state.configurator = ComputerConfigurator()
    
    init_session_state()

    # Initialize configurator
    # configurator = st.session_state.configurator

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Display available tasks at start
    if len(st.session_state.messages) == 0:
        tasks = list(configurator.task_requirements.keys())
        task_list = "\n".join([f"{i}. {task}" for i, task in enumerate(tasks, 1)])
        system_msg = f"Welcome! Please enter your desired task from the following options:\n\n{task_list}\n\nOr press Enter for a general build."
        add_message("assistant", system_msg)

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        add_message("user", prompt)

        # Handle user input based on current step
        if st.session_state.current_step == 'task':
            is_valid, validated_task, suggestions = configurator.input_handler.validate_task(prompt)
            
            if is_valid:
                st.session_state.task = validated_task
                st.session_state.current_step = 'budget'
                add_message("assistant", f"Task set to: {validated_task}\nPlease enter your budget (or press Enter for unlimited):")
            else:
                suggestion_list = "\n".join([f"- {s}" for s in suggestions])
                add_message("assistant", f"Invalid task. Did you mean one of these?\n{suggestion_list}")

        elif st.session_state.current_step == 'budget':
            is_valid, validated_budget = configurator.input_handler.validate_budget(prompt)
            
            if is_valid:
                st.session_state.budget = validated_budget
                
                # Generate configuration
                configurator.reset()
                configurator.declare(UserPreferences(
                    task=st.session_state.task, 
                    budget=validated_budget
                ))
                configurator.run()

                # Format and display results
                if configurator.best_config:
                    best_config = configurator.format_configuration(
                        configurator.best_config, 
                        include_explanations=True
                    )
                    add_message("assistant", f"Best Configuration:\n```\n{best_config}\n```")

                    # Show alternatives
                    for i, alt in enumerate(configurator.alternative_configs[:3], 1):
                        alt_config = configurator.format_configuration(alt)
                        add_message("assistant", f"Alternative {i}:\n```\n{alt_config}\n```")
                else:
                    add_message("assistant", "No valid configurations found for your requirements.")

                # Reset for new configuration
                st.session_state.current_step = 'task'
                st.session_state.task = None
                st.session_state.budget = None
                
            else:
                add_message("assistant", "Invalid budget. Please enter a positive number.")

    # Add reset button
    if st.button("Start New Configuration"):
        st.session_state.messages = []
        st.session_state.current_step = 'task'
        st.session_state.task = None
        st.session_state.budget = None
        st.rerun()

if __name__ == "__main__":
    main()

# def main():
#     st.title("Computer Configuration Expert System")
    
#     # Initialize expert system
#     configurator = ComputerConfigurator()
#     input_handler = configurator.input_handler
    
#     # Task selection
#     task = st.text_input("Enter task (or leave empty for general build)")
#     is_valid, validated_task, suggestions = input_handler.validate_task(task)
    
#     if not is_valid and suggestions:
#         st.warning("Did you mean one of these?")
#         selected_task = st.selectbox("Select task:", suggestions)
#         validated_task = selected_task
#     elif not task:
#         validated_task = input_handler.DEFAULT_TASK
#         st.info(f"Using default task: {validated_task}")
    
#     # Budget input
#     budget_input = st.text_input("Enter budget (or leave empty for unlimited)")
#     is_valid, validated_budget = input_handler.validate_budget(budget_input)
    
#     if not is_valid:
#         st.error("Please enter a valid positive number for budget")
#     elif not budget_input:
#         validated_budget = input_handler.DEFAULT_BUDGET
#         st.info("Using unlimited budget")
    
#     # Configure button
#     if st.button("Configure System"):
#         if validated_task and validated_budget:
#             configurator.reset()
#             configurator.declare(UserPreferences(task=validated_task, budget=validated_budget))
#             configurator.run()
            
#             # Display results
#             if configurator.best_config:
#                 st.subheader("Best Configuration")
#                 st.text(configurator.format_configuration(configurator.best_config, include_explanations=True))
                
#                 st.subheader("Alternative Configurations")
#                 for i, alt in enumerate(configurator.alternative_configs[:3], 1):
#                     st.text(f"\nAlternative {i}:")
#                     st.text(configurator.format_configuration(alt))

# if __name__ == "__main__":
#     main()