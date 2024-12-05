import json
from experta import *
from typing import List, Dict
import difflib
import os


# class ComponentSelection(Fact):
#     """Fact for storing selected component and its score."""
#     pass


class UserPreferences(Fact):
    """Fact to capture user inputs."""
    pass


class Configuration(Fact):
    """Fact to store complete PC configurations"""
    pass


class ComputerConfigurator(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.load_knowledge_base()
        self.alternative_configs = []  # Initialize empty list for alternative configs
        self.best_config = None  # Also initialize best_config as None
        self.seen_configs = set()  # Track unique configurations

    def load_knowledge_base(self):
        print(os.getcwd())
        with open('compfigurator\\knowledge_base.json', 'r') as file:
            data = json.load(file)
            self.task_requirements = data['taskRequirements']
            self.components = data['components']

    @DefFacts()
    def initial_facts(self):
        yield Fact(action="start")

    # Step 1 helper function

    def get_closest_match(self, user_input, options):
        matches = difflib.get_close_matches(
            user_input, options, n=1, cutoff=0.1)
        return matches[0] if matches else None

    @Rule(
        Fact(action='start'),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def select_processors(self, task, budget):
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_cpu_budget = budget * budget_alloc['cpu']

        # Consider all processors under total budget instead of cpu allocation
        processors = self.components['processors']
        suitable_processors = [
            p for p in processors if p['price'] <= budget]

        # Sort by how close they are to suggested budget
        suitable_processors.sort(key=lambda p: abs(
            p['price'] - suggested_cpu_budget))

        for processor in suitable_processors:
            self.declare(Fact(component_type="cpu",
                         name=processor["name"], price=processor["price"]))

    @Rule(
        Fact(component_type="cpu", name=MATCH.cpu_name, price=MATCH.cpu_price),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_motherboards(self, cpu_name, cpu_price, task, budget):
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_mb_budget = budget * budget_alloc['mb']

        # Get processor details from knowledge base
        processor = next(
            p for p in self.components["processors"] if p["name"] == cpu_name)

        # Find compatible motherboards under total remaining budget
        remaining_budget = budget - cpu_price
        compatible_motherboards = [
            mb for mb in self.components["motherboards"]
            if (mb["socket"] == processor["socket"] and
                any(chipset in processor["supportedChipsets"] for chipset in [mb["chipset"]]) and
                mb["price"] <= remaining_budget)
        ]

        # Sort by closeness to suggested budget
        compatible_motherboards.sort(
            key=lambda mb: abs(mb["price"] - suggested_mb_budget))

        # Declare compatible motherboards as facts
        for mb in compatible_motherboards:
            self.declare(
                Fact(
                    component_type="motherboard",
                    name=mb["name"],
                    price=mb["price"],
                    paired_cpu=cpu_name,
                    socket=mb["socket"],
                    chipset=mb["chipset"]
                )
            )

    @Rule(
        Fact(component_type="motherboard",
             name=MATCH.mb_name,
             price=MATCH.mb_price,
             paired_cpu=MATCH.cpu_name,
             socket=MATCH.socket,
             chipset=MATCH.chipset),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_gpus(self, mb_name, mb_price, cpu_name, task, budget):
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_gpu_budget = budget * budget_alloc['gpu']

        # Get previously selected components' prices
        cpu_price = next(
            p["price"] for p in self.components["processors"] if p["name"] == cpu_name)
        remaining_budget = budget - (cpu_price + mb_price)

        # Get motherboard details
        motherboard = next(
            mb for mb in self.components["motherboards"] if mb["name"] == mb_name)

        # Find compatible GPUs based on PCIe slots and budget
        compatible_gpus = [
            gpu for gpu in self.components["gpus"]
            if (gpu["requiredPcieSlots"] <= motherboard["pciSlots"] and
                gpu["price"] <= remaining_budget)
        ]

        # Sort by closeness to suggested budget
        compatible_gpus.sort(key=lambda gpu: abs(
            gpu["price"] - suggested_gpu_budget))

        # Declare compatible GPUs as facts
        for gpu in compatible_gpus:
            self.declare(
                Fact(
                    component_type="gpu",
                    name=gpu["name"],
                    price=gpu["price"],
                    paired_cpu=cpu_name,
                    paired_mb=mb_name,
                    vram=gpu["vram"]
                )
            )

    @Rule(
        Fact(component_type="motherboard",
             name=MATCH.mb_name,
             price=MATCH.mb_price,
             paired_cpu=MATCH.cpu_name),
        Fact(component_type="gpu",
             name=MATCH.gpu_name,
             price=MATCH.gpu_price),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_ram(self, mb_name, mb_price, cpu_name, gpu_name, gpu_price, task, budget):
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_ram_budget = budget * budget_alloc['ram']

        # Get previous components' prices
        cpu_price = next(
            p["price"] for p in self.components["processors"] if p["name"] == cpu_name)
        remaining_budget = budget - (cpu_price + mb_price + gpu_price)

        # Get motherboard and CPU details
        motherboard = next(
            mb for mb in self.components["motherboards"] if mb["name"] == mb_name)
        cpu = next(
            cpu for cpu in self.components["processors"] if cpu["name"] == cpu_name)

        # Find compatible RAM based on DDR version, speed, and channels
        compatible_ram = [
            ram for ram in self.components["ram"]
            if (ram["type"] == motherboard["supportedDDR"] and
                ram["speed"] <= cpu["maxMemorySpeed"] and
                ram["price"] <= remaining_budget)
        ]

        compatible_ram.sort(key=lambda ram: abs(
            ram["price"] - suggested_ram_budget))

        # Declare compatible RAM as facts
        for ram in compatible_ram:
            self.declare(
                Fact(
                    component_type="ram",
                    name=ram["name"],
                    price=ram["price"],
                    paired_cpu=cpu_name,
                    paired_mb=mb_name,
                    capacity=ram["capacity"],
                    speed=ram["speed"]
                )
            )

    @Rule(
        Fact(component_type="motherboard",
             name=MATCH.mb_name,
             price=MATCH.mb_price,
             paired_cpu=MATCH.cpu_name),
        Fact(component_type="gpu",
             name=MATCH.gpu_name,
             price=MATCH.gpu_price),
        Fact(component_type="ram",
             name=MATCH.ram_name,
             price=MATCH.ram_price),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_ssd(self, mb_name, mb_price, cpu_name, gpu_name, gpu_price, ram_name, ram_price, task, budget):
        # Calculate suggested SSD budget
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_ssd_budget = budget * budget_alloc['ssd']

        # Get previous components' prices
        cpu_price = next(
            p["price"] for p in self.components["processors"] if p["name"] == cpu_name)
        remaining_budget = budget - \
            (cpu_price + mb_price + gpu_price + ram_price)

        # Get motherboard details
        motherboard = next(
            mb for mb in self.components["motherboards"] if mb["name"] == mb_name)

        # Find compatible SSDs based on interface and budget
        compatible_ssds = [
            ssd for ssd in self.components["storage"]
            if (ssd["interface"] in motherboard["storageInterfaces"] and
                ssd["price"] <= remaining_budget)
        ]

        # Sort by closeness to suggested budget
        compatible_ssds.sort(key=lambda ssd: abs(
            ssd["price"] - suggested_ssd_budget))

        # Declare compatible SSDs as facts
        for ssd in compatible_ssds:
            self.declare(
                Fact(
                    component_type="storage",
                    name=ssd["name"],
                    price=ssd["price"],
                    paired_cpu=cpu_name,
                    paired_mb=mb_name,
                    capacity=ssd["capacity"],
                    interface=ssd["interface"]
                )
            )

    @Rule(
        Fact(component_type="gpu",
             name=MATCH.gpu_name,
             price=MATCH.gpu_price,
             paired_cpu=MATCH.cpu_name,
             paired_mb=MATCH.mb_name),
        Fact(component_type="storage",
             name=MATCH.storage_name,
             price=MATCH.storage_price),
        Fact(component_type="ram",
             name=MATCH.ram_name,
             price=MATCH.ram_price),
        Fact(component_type="motherboard",
             name=MATCH.mb_name,
             price=MATCH.mb_price,
             paired_cpu=MATCH.cpu_name),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_monitors(self, gpu_name, storage_name, ram_name, mb_name,
                                 gpu_price, storage_price, ram_price, mb_price, cpu_name, task, budget):
        # Calculate suggested monitor budget
        budget_alloc = self.task_requirements[task]['budgetAlloc']
        suggested_monitor_budget = budget * budget_alloc['monitor']

        # Get previous components' prices
        cpu_price = next(
            p["price"] for p in self.components["processors"] if p["name"] == cpu_name)
        remaining_budget = budget - \
            (cpu_price + mb_price + gpu_price + ram_price + storage_price)

        # Get GPU details for resolution/refresh rate compatibility
        gpu = next(g for g in self.components["gpus"] if g["name"] == gpu_name)

        # Find suitable monitors within budget
        suitable_monitors = [
            monitor for monitor in self.components["monitors"]
            if (monitor["price"] <= remaining_budget)
        ]

        # Sort by closeness to suggested budget
        suitable_monitors.sort(key=lambda m: abs(
            m["price"] - suggested_monitor_budget))

        # Declare suitable monitors as facts
        for monitor in suitable_monitors:
            self.declare(
                Fact(
                    component_type="monitor",
                    name=monitor["name"],
                    price=monitor["price"],
                    paired_gpu=gpu_name,
                    paired_cpu=cpu_name,
                    paired_mb=mb_name,
                    resolution=monitor["resolution"],
                    refresh_rate=monitor["refreshRate"],
                    panel_type=monitor["panelType"]
                )
            )

    @Rule(
        Fact(component_type="cpu", name=MATCH.cpu_name, price=MATCH.cpu_price),
        Fact(component_type="motherboard", name=MATCH.mb_name,
             price=MATCH.mb_price, paired_cpu=MATCH.cpu_name),
        Fact(component_type="gpu", name=MATCH.gpu_name, price=MATCH.gpu_price,
             paired_cpu=MATCH.cpu_name, paired_mb=MATCH.mb_name),
        Fact(component_type="ram", name=MATCH.ram_name, price=MATCH.ram_price,
             paired_cpu=MATCH.cpu_name, paired_mb=MATCH.mb_name),
        Fact(component_type="storage", name=MATCH.storage_name,
             price=MATCH.storage_price, paired_cpu=MATCH.cpu_name, paired_mb=MATCH.mb_name),
        Fact(component_type="monitor", name=MATCH.monitor_name,
             price=MATCH.monitor_price, paired_gpu=MATCH.gpu_name),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def generate_configuration(self, cpu_name, mb_name, gpu_name, ram_name, storage_name, monitor_name,
                               cpu_price, mb_price, gpu_price, ram_price, storage_price, monitor_price,
                               task, budget):

        # Calculate total price
        total_price = sum([cpu_price, mb_price, gpu_price,
                          ram_price, storage_price, monitor_price])

        # Only proceed if total price is within budget
        if total_price <= budget:
            config = {
                "cpu": cpu_name,
                "motherboard": mb_name,
                "gpu": gpu_name,
                "ram": ram_name,
                "storage": storage_name,
                "monitor": monitor_name,
                "total": total_price  # Add total price to configuration
            }

            score = self.calculate_configuration_score(config, task)

            self.declare(Configuration(
                components=config,
                score=score,
                total_price=total_price,  # Add total price to fact
                task=task
            ))

    def calculate_configuration_score(self, config, task):
        # Get max values
        cpu_max, gpu_max, ram_max = self.get_max_values()

        # Get component details
        cpu = next(
            c for c in self.components["processors"] if c["name"] == config["cpu"])
        gpu = next(
            g for g in self.components["gpus"] if g["name"] == config["gpu"])
        ram = next(
            r for r in self.components["ram"] if r["name"] == config["ram"])

        # Calculate normalized performance scores
        cpu_perf = (
            (cpu["cores"] / cpu_max['cores'] * 0.3) +
            (cpu["baseSpeed"] / cpu_max['baseSpeed'] * 0.3) +
            (cpu["boostSpeed"] / cpu_max['boostSpeed'] * 0.4)
        )

        gpu_perf = (
            (gpu["vram"] / gpu_max['vram'] * 0.4) +
            (gpu["clockSpeed"] / gpu_max['clockSpeed'] * 0.3) +
            (gpu["cudaCores"] / gpu_max['cudaCores'] * 0.3)
        )

        ram_perf = (
            (ram["capacity"] / ram_max['capacity'] * 0.6) +
            (ram["speed"] / ram_max['speed'] * 0.4)
        )

        # Weight performances by task requirements
        task_reqs = self.task_requirements[task]
        score = (
            cpu_perf * task_reqs["budgetAlloc"]["cpu"] +
            gpu_perf * task_reqs["budgetAlloc"]["gpu"] +
            ram_perf * task_reqs["budgetAlloc"]["ram"]
        )
        return score

    def get_max_values(self):
        # CPU max values
        cpu_max = {
            'cores': max(cpu['cores'] for cpu in self.components['processors']),
            'baseSpeed': max(cpu['baseSpeed'] for cpu in self.components['processors']),
            'boostSpeed': max(cpu['boostSpeed'] for cpu in self.components['processors'])
        }

        # GPU max values
        gpu_max = {
            'vram': max(gpu['vram'] for gpu in self.components['gpus']),
            'clockSpeed': max(gpu['clockSpeed'] for gpu in self.components['gpus']),
            'cudaCores': max(gpu['cudaCores'] for gpu in self.components['gpus'])
        }

        # RAM max values
        ram_max = {
            'capacity': max(ram['capacity'] for ram in self.components['ram']),
            'speed': max(ram['speed'] for ram in self.components['ram'])
        }

        return cpu_max, gpu_max, ram_max

    def config_to_tuple(self, config):
        """Convert configuration to hashable tuple for uniqueness checking"""
        return tuple(sorted([
            (k, v) for k, v in config['components'].items()
        ]))

    @Rule(AS.config << Configuration())
    def select_best_configuration(self, config):
        config_data = {
            "score": config["score"],
            "components": config["components"],
            "total_price": config["total_price"],
            "task": config["task"]
        }

        # Create unique identifier for this configuration
        config_tuple = self.config_to_tuple(config_data)

        # Only process if we haven't seen this configuration before
        if config_tuple not in self.seen_configs:
            self.seen_configs.add(config_tuple)

            if (self.best_config is None or
                    config_data["score"] > self.best_config["score"]):
                # If current config is better, move existing best to alternatives
                if self.best_config is not None:
                    self.alternative_configs.append(self.best_config)
                self.best_config = config_data
            else:
                self.alternative_configs.append(config_data)

    def get_component_explanation(self, component_type, component_name, task):
        # Get component from knowledge base
        component = None
        if component_type == "cpu":
            component = next(
                c for c in self.components["processors"] if c["name"] == component_name)
        elif component_type == "gpu":
            component = next(
                g for g in self.components["gpus"] if g["name"] == component_name)
        elif component_type == "motherboard":
            component = next(
                m for m in self.components["motherboards"] if m["name"] == component_name)
        elif component_type == "ram":
            component = next(
                r for r in self.components["ram"] if r["name"] == component_name)
        elif component_type == "storage":
            component = next(
                s for s in self.components["storage"] if s["name"] == component_name)
        elif component_type == "monitor":
            component = next(
                m for m in self.components["monitors"] if m["name"] == component_name)

        # Get explanation
        if component and "explain" in component:
            return component["explain"].get(task, component["explain"]["default"])
        return ""

    def format_configuration(self, config, include_explanations=False):
        if not config:  # If config is None or empty
            return "No configuration available"
        try:
            cpu = next(
                c for c in self.components["processors"] if c["name"] == config["components"]["cpu"])
            mb = next(m for m in self.components["motherboards"]
                      if m["name"] == config["components"]["motherboard"])
            gpu = next(
                g for g in self.components["gpus"] if g["name"] == config["components"]["gpu"])
            ram = next(
                r for r in self.components["ram"] if r["name"] == config["components"]["ram"])
            storage = next(
                s for s in self.components["storage"] if s["name"] == config["components"]["storage"])
            monitor = next(
                m for m in self.components["monitors"] if m["name"] == config["components"]["monitor"])

            task = config["task"]

            if include_explanations:
                cpu_explain = self.get_component_explanation(
                    "cpu", cpu["name"], task)
                mb_explain = self.get_component_explanation(
                    "motherboard", mb["name"], task)
                gpu_explain = self.get_component_explanation(
                    "gpu", gpu["name"], task)
                ram_explain = self.get_component_explanation(
                    "ram", ram["name"], task)
                storage_explain = self.get_component_explanation(
                    "storage", storage["name"], task)
                monitor_explain = self.get_component_explanation(
                    "monitor", monitor["name"], task)

                return f"""
                    Configuration Score: {config['score']:.2f}
                    Total Price: ${config['total_price']:.2f}
                    Components:
                    - CPU: {config['components']['cpu']} (${cpu['price']:.2f}) because {cpu_explain}
                    - Motherboard: {config['components']['motherboard']} (${mb['price']:.2f}) because {mb_explain}
                    - GPU: {config['components']['gpu']} (${gpu['price']:.2f}) because {gpu_explain}
                    - RAM: {config['components']['ram']} (${ram['price']:.2f}) because {ram_explain}
                    - Storage: {config['components']['storage']} (${storage['price']:.2f}) because {storage_explain}
                    - Monitor: {config['components']['monitor']} (${monitor['price']:.2f}) because {monitor_explain}
                """
            else:
                return f"""
                    Configuration Score: {config['score']:.2f}
                    Total Price: ${config['total_price']:.2f}
                    Components:
                    - CPU: {config['components']['cpu']} (${cpu['price']:.2f})
                    - Motherboard: {config['components']['motherboard']} (${mb['price']:.2f})
                    - GPU: {config['components']['gpu']} (${gpu['price']:.2f})
                    - RAM: {config['components']['ram']} (${ram['price']:.2f})
                    - Storage: {config['components']['storage']} (${storage['price']:.2f})
                    - Monitor: {config['components']['monitor']} (${monitor['price']:.2f})
                """
        except (KeyError, StopIteration):
            return "Invalid configuration data"
        except Exception as e:
            return f"An error occurred: {e}"

    def print_configurations(self):
        print("\n=== BEST CONFIGURATION ===")

        if self.best_config:
            print(self.format_configuration(
                self.best_config, include_explanations=True))
        else:
            print("No configuration available for the given task and budget.")

        print("\n=== ALTERNATIVE CONFIGURATIONS ===")
        # Sort alternatives by score in descending order

        sorted_alts = sorted(self.alternative_configs,
                             key=lambda x: x['score'],
                             reverse=True)[:3]  # Show top 3 alternatives
        if not sorted_alts:
            print("No alternative configurations available")
        else:
            for i, alt in enumerate(sorted_alts, 1):
                print(f"\nAlternative {i}:")
                print(self.format_configuration(alt))

            # Print "No configuration available" for missing alternatives
            for i in range(len(sorted_alts) + 1, 4):
                print(f"\nAlternative {i}:")
                print("No configuration available")


if __name__ == "__main__":
    engine = ComputerConfigurator()
    engine.reset()
    engine.declare(UserPreferences(task="machineLearning", budget=3000))
    # engine.declare(UserPreferences(task="gaming", budget=20000))
    engine.run()
    engine.print_configurations()

"""

# **Key Features in This Version**
1. ** Command-Line Input**: Collects task and budget directly from the user.
2. ** Explanations**: Each selected component is explained clearly.
3. ** Uncertainty Handling**: Suitability scores are calculated for components based on task requirements.
4. ** Alternatives**: Displays alternative solutions with suitability scores.
5. ** Budget Flexibility**: Can be incorporated by evaluating suitability vs. budget trade-offs dynamically.

You can extend this CLI to a GUI using frameworks like ** Streamlit ** or **Tkinter**.

"""
