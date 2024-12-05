import json
from experta import *
from typing import List, Dict
import difflib
import os
import socket
import threading


class ComponentSelection(Fact):
    """Fact for storing selected component and its score."""
    pass


class UserPreferences(Fact):
    """Fact to capture user inputs."""
    pass


class BudgetAllocation(Fact):
    """Fact to capture budget allocation for each component type."""
    pass


class ComputerConfigurator(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.load_knowledge_base()
        self.recommended_solution = None
        self.alternative_solutions = []

    def load_knowledge_base(self):
        with open('knowledge_base.json', 'r') as file:
            data = json.load(file)
            self.task_requirements = data['task_requirements']
            self.components = data['components']

    @DefFacts()
    def initial_facts(self):
        yield Fact(action="start")

    def get_closest_match(self, user_input, options):
        matches = difflib.get_close_matches(
            user_input, options, n=1, cutoff=0.1)
        return matches[0] if matches else None

    def ask_user_input(self, task, budget):
        self.declare(UserPreferences(task=task, budget=budget))
        self.run()

    @Rule(Fact(action="start"))
    def get_user_preferences(self):
        self.declare(Fact(action="allocate_budget"))

    @Rule(Fact(action="allocate_budget"), UserPreferences(task=MATCH.task, budget=MATCH.budget))
    def allocate_budget(self, task, budget):
        allocation = self.calculate_budget_allocation(task, budget)
        self.declare(BudgetAllocation(**allocation))
        self.declare(Fact(action="recommend_components"))

    def calculate_budget_allocation(self, task, budget):
        allocation = {
            "processor": 0.3 * budget,
            "ram": 0.2 * budget,
            "vga": 0.3 * budget,
            "motherboard": 0.1 * budget,
            "ssd": 0.05 * budget,
            "monitor": 0.05 * budget
        }
        if task == "ml":
            allocation["processor"] = 0.35 * budget
            allocation["vga"] = 0.35 * budget
            allocation["ram"] = 0.15 * budget
        elif task == "graphic design" or task == "video editing":
            allocation["monitor"] = 0.2 * budget
            allocation["vga"] = 0.3 * budget
            allocation["processor"] = 0.25 * budget
        elif task == "general":
            allocation["processor"] = 0.25 * budget
            allocation["ram"] = 0.25 * budget
            allocation["ssd"] = 0.2 * budget
        return allocation

    def find_best_component(self, component_type, task, budget):
        best_component = None
        best_score = 0
        alternatives = []

        for component in self.components[component_type]:
            if component["cost"] <= budget:
                score = self.calculate_suitability(task, component["power"])
                if score > best_score:
                    best_component = component
                    best_score = score
                alternatives.append((component, score))
        return best_component, alternatives

    @Rule(Fact(action="recommend_components"), UserPreferences(task=MATCH.task, budget=MATCH.budget), BudgetAllocation(processor=MATCH.processor_budget, ram=MATCH.ram_budget, vga=MATCH.vga_budget, motherboard=MATCH.motherboard_budget, ssd=MATCH.ssd_budget, monitor=MATCH.monitor_budget))
    def recommend_system(self, task, budget, processor_budget, ram_budget, vga_budget, motherboard_budget, ssd_budget, monitor_budget):
        processor, processor_alts = self.find_best_component(
            "processor", task, processor_budget)
        motherboard, motherboard_alts = self.find_best_component(
            "motherboard", task, motherboard_budget)
        ram, ram_alts = self.find_best_component("ram", task, ram_budget)
        vga, vga_alts = self.find_best_component("vga", task, vga_budget)
        ssd, ssd_alts = self.find_best_component("ssd", task, ssd_budget)
        monitor, monitor_alts = self.find_best_component(
            "monitor", task, monitor_budget)

        self.recommended_solution = {
            "processor": processor,
            "motherboard": motherboard,
            "ram": ram,
            "vga": vga,
            "ssd": ssd,
            "monitor": monitor
        }
        self.alternative_solutions = {
            "processors": processor_alts,
            "motherboards": motherboard_alts,
            "ram": ram_alts,
            "vga": vga_alts,
            "ssd": ssd_alts,
            "monitor": monitor_alts
        }

        self.explain_recommendation()

    def explain_recommendation(self):
        response = "\nRecommended System Configuration:\n"
        total_cost = 0
        for comp_type, component_with_score in self.recommended_solution.items():
            if component_with_score is None:
                continue
            component = component_with_score
            total_cost += component["cost"]
            response += f"{comp_type.capitalize()}: {component['name']} (${
                component['cost']}) because {component['explanation']}\n"
        response += f"Total Cost: ${total_cost}\n"

        response += "\nAlternative Solutions:\n"
        for comp_type, alternatives in self.alternative_solutions.items():
            response += f"{comp_type.capitalize()}:\n"
            for alt, score in alternatives:
                response += f" - {alt['name']
                                  } (${alt['cost']}) [Suitability: {score}]\n"

        return response

    def show_components(self):
        response = "\nAvailable Components:\n"
        for comp_type, components in self.components.items():
            response += f"\n{comp_type.capitalize()}:\n"
            for component in components:
                response += f" - {component['name']} (${component['cost']}): {
                    component['explanation']}\n"

        return response


def handle_client(client_socket, engine):
    while True:
        request = client_socket.recv(1024).decode('utf-8')
        if not request:
            break

        if request.startswith("CONFIGURE"):
            _, task, budget = request.split()
            budget = int(budget)
            engine.ask_user_input(task, budget)
            response = engine.explain_recommendation()
        elif request.startswith("VIEW_COMPONENTS"):
            response = engine.show_components()
        else:
            response = "Invalid command."

        client_socket.send(response.encode('utf-8'))

    client_socket.close()


def start_server():
    engine = ComputerConfigurator()
    engine.reset()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 9999))
    server.listen(5)
    print("Server listening on port 9999")

    while True:
        client_socket, addr = server.accept()
        client_handler = threading.Thread(
            target=handle_client, args=(client_socket, engine))
        client_handler.start()


if __name__ == "__main__":
    start_server()
