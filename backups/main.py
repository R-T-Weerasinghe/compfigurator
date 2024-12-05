import json
from experta import *
from typing import List, Dict
import difflib
import os

class ComponentSelection(Fact):
    """Fact for storing selected component and its score."""
    pass


class UserPreferences(Fact):
    """Fact to capture user inputs."""
    pass


class ComputerConfigurator(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.load_knowledge_base()
        self.recommended_solution = None
        self.alternative_solutions = []


    def load_knowledge_base(self):
        print(os.getcwd())
        with open('compfigurator\\knowledge_base.json', 'r') as file:
            data = json.load(file)
            self.task_requirements = data['task_requirements']
            self.components = data['components']


    @DefFacts()
    def initial_facts(self):
        yield Fact(action="start")


    # Step 1 helper function
    def get_closest_match(self, user_input, options):
        matches = difflib.get_close_matches(user_input, options, n=1, cutoff=0.1)
        return matches[0] if matches else None
    

    # Step 1: Ask user for task and budget
    def ask_user_input(self):
        while True:
            task = input("Enter your task (e.g., moderate_ml): ")
            if task in self.task_requirements:
                break
            else:
                closest_match = self.get_closest_match(task, self.task_requirements.keys())
                if closest_match:
                    response = input(f"Did you mean '{closest_match}'? (yes/no): ").strip().lower()
                    if response.strip().lower() == 'yes':
                        task = closest_match
                        break
                    elif response.strip().lower() == 'no':
                        print("Please enter the task again.")
                    else:
                        print("I can't understand. Please answer with 'yes' or 'no'.")
                else:
                    print("No close match found. Please enter the task again.")

        while True:
            try:
                raw = input("Enter your budget (in USD): ")
                # IF budget is not given, assume user has unlimited budget
                if raw is None:
                    raw = "9999999"
                budget = int(raw)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number for the budget.")

        self.declare(UserPreferences(task=task, budget=budget)) # then goes to step 0 again


    # Step 0: Start the process
    @Rule(Fact(action="start"))
    def get_user_preferences(self):
        self.ask_user_input()
        self.declare(Fact(action="recommend_components"))


    # TODO: rethinking the logic
    def calculate_suitability(self, task, component_power):
        """Calculate suitability score based on task requirements."""
        return 0.5
        for device in self.task_requirements[task].keys():
            pass
        for power, score in self.task_requirements[task].get(component_power, []):
            if power == component_power:
                return score
        return 0
    

    def find_best_component(self, component_type, task, budget):
        best_component = None
        best_score = 0
        alternatives = []

        for component in self.components[component_type]:
            if component["cost"] <= budget:
                print(component["power"])
                score = self.calculate_suitability(task, component["power"])
                if score > best_score:
                    best_component = component
                    best_score = score
                alternatives.append((component, score))
                # alternatives.append((component, 0.5))  # Remove this line
            # best_component = alternatives[0]    # Remove this line
        return best_component, alternatives

    # Step 2: Recommend components
    @Rule(Fact(action="recommend_components"), UserPreferences(task=MATCH.task, budget=MATCH.budget))
    def recommend_system(self, task, budget):
        # Select processor
        processor, processor_alts = self.find_best_component(
            "processor", task, budget)
        motherboard, motherboard_alts = self.find_best_component(
            "motherboard", task, budget)
        ram, ram_alts = self.find_best_component("ram", task, budget)
        vga, vga_alts = self.find_best_component("vga", task, budget)

        # Calculate overall suitability and add alternatives
        self.recommended_solution = {
            "processor": processor,
            "motherboard": motherboard,
            "ram": ram,
            "vga": vga
        }
        self.alternative_solutions = {
            "processors": processor_alts,
            "motherboards": motherboard_alts,
            "ram": ram_alts,
            "vga": vga_alts
        }

        self.explain_recommendation()

    def explain_recommendation(self):
        print("\nRecommended System Configuration:")
        # print(self.recommended_solution)
        total_cost = 0
        # for key, value in self.recommended_solution.items():
        #     print(f"{key.capitalize()}: {value}")
        for comp_type, component_with_score in self.recommended_solution.items():
            # None check
            if component_with_score is None:
                continue
            component = component_with_score
            total_cost += component["cost"]
            print(f"{comp_type.capitalize()}: {component['name']} (${
                  component['cost']}) because {component['explanation']}")
        # total_cost = sum([comp["cost"]
        #                  for comp in self.recommended_solution.values()])
        # for comp_type, component in self.recommended_solution.items():
        #     print(f"{comp_type.capitalize()}: {component['name']} (${
        #           component['cost']}) - {component['explanation']}")
        print(f"Total Cost: ${total_cost}")

        print("\nAlternative Solutions:")
        for comp_type, alternatives in self.alternative_solutions.items():
            print(f"{comp_type.capitalize()}:")
            for alt, score in alternatives:
                print(
                    f" - {alt['name']} (${alt['cost']}) [Suitability: {score}]")

    def show_components(self):
        print("\nAvailable Components:")
        for comp_type, components in self.components.items():
            print(f"\n{comp_type.capitalize()}:")
            for component in components:
                print(f" - {component['name']} (${component['cost']}): {component['explanation']}")

    # ----------------
    # UI Codes
    # ----------------
    def main_menu(self):
        while True:
            print("\nWelcome to the Computer Configurator!")
            print("1: Configure")
            print("2: View components")
            print("e: Exit")
            choice = input("Please select an option: ").strip().lower()

            if choice == '1':
                self.configure_menu()
            elif choice == '2':
                self.view_components_menu()
            elif choice == 'e':
                print("Exiting the application. Goodbye!")
                break
            else:
                print("Invalid option. Please try again.")

    def configure_menu(self):
        while True:
            print("\nConfiguration Menu")
            print("Type 'back' to return to the main menu.")
            self.ask_user_input()
            print("Configuration complete. Returning to main menu.")
            break

    def view_components_menu(self):
        while True:
            print("--------------------------------")
            print("\nView Components Menu")
            print("--------------------------------")
            self.show_components()
            choice = input("Press Enter to go back to the main menu: ").strip().lower()
            if choice == 'back' or choice == '':
                break

if __name__ == "__main__":
    engine = ComputerConfigurator()
    engine.reset()  # Prepare the engine for running (initializes facts)
    engine.run()    # Start the rules engine
    # engine.main_menu()  # Start the main menu

"""

# **Key Features in This Version**
1. ** Command-Line Input**: Collects task and budget directly from the user.
2. ** Explanations**: Each selected component is explained clearly.
3. ** Uncertainty Handling**: Suitability scores are calculated for components based on task requirements.
4. ** Alternatives**: Displays alternative solutions with suitability scores.
5. ** Budget Flexibility**: Can be incorporated by evaluating suitability vs. budget trade-offs dynamically.

You can extend this CLI to a GUI using frameworks like ** Streamlit ** or **Tkinter**.

"""