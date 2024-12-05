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


class Configuration(Fact):
    """Fact to store complete PC configurations"""
    pass


class ComputerConfigurator(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.load_knowledge_base()
        self.alternative_configs = []  # Initialize empty list for alternative configs
        self.best_config = None  # Also initialize best_config as None

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
        cpu_budget = budget * budget_alloc['cpu']

        processors = self.components['processors']
        suitable_processors = [
            p for p in processors if p['price'] <= cpu_budget]

        for processor in suitable_processors:
            self.declare(Fact(component_type="cpu",
                         name=processor["name"], price=processor["price"]))

    @Rule(
        Fact(component_type="cpu", name=MATCH.cpu_name, price=MATCH.cpu_price),
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_motherboards(self, cpu_name, cpu_price, task, budget):
        # Get processor details from knowledge base
        processor = next(
            p for p in self.components["processors"] if p["name"] == cpu_name)

        # Calculate motherboard budget
        mb_budget = budget * self.task_requirements[task]["budgetAlloc"]["mb"]

        # Find compatible motherboards
        compatible_motherboards = [
            mb for mb in self.components["motherboards"]
            if (mb["socket"] == processor["socket"] and
                any(chipset in processor["supportedChipsets"] for chipset in [mb["chipset"]]) and
                mb["price"] <= mb_budget)
        ]

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
        # Calculate GPU budget
        gpu_budget = budget * \
            self.task_requirements[task]["budgetAlloc"]["gpu"]

        # Get motherboard details
        motherboard = next(
            mb for mb in self.components["motherboards"] if mb["name"] == mb_name)

        # Find compatible GPUs based on PCIe slots and budget
        compatible_gpus = [
            gpu for gpu in self.components["gpus"]
            if (gpu["requiredPcieSlots"] <= motherboard["pciSlots"] and
                gpu["price"] <= gpu_budget)
        ]

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
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_ram(self, mb_name, mb_price, cpu_name, task, budget):
        # Calculate RAM budget
        ram_budget = budget * \
            self.task_requirements[task]["budgetAlloc"]["ram"]

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
                ram["price"] <= ram_budget)
        ]

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
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_ssd(self, mb_name, mb_price, cpu_name, task, budget):
        # Calculate SSD budget
        ssd_budget = budget * \
            self.task_requirements[task]["budgetAlloc"]["ssd"]

        # Get motherboard details
        motherboard = next(
            mb for mb in self.components["motherboards"] if mb["name"] == mb_name)

        # Find compatible SSDs based on interface and budget
        compatible_ssds = [
            ssd for ssd in self.components["storage"]
            if (ssd["interface"] in motherboard["storageInterfaces"] and
                ssd["price"] <= ssd_budget)
        ]

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
        UserPreferences(task=MATCH.task, budget=MATCH.budget)
    )
    def find_compatible_monitors(self, gpu_name, gpu_price, cpu_name, mb_name, task, budget):
        # Calculate monitor budget
        monitor_budget = budget * \
            self.task_requirements[task]["budgetAlloc"]["monitor"]

        # Get GPU details for resolution/refresh rate compatibility
        gpu = next(g for g in self.components["gpus"] if g["name"] == gpu_name)

        # Find suitable monitors within budget
        suitable_monitors = [
            monitor for monitor in self.components["monitors"]
            if (monitor["price"] <= monitor_budget)
        ]

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

        total_price = cpu_price + mb_price + gpu_price + \
            ram_price + storage_price + monitor_price

        if total_price <= budget:
            # Create configuration object
            config = {
                "cpu": cpu_name,
                "motherboard": mb_name,
                "gpu": gpu_name,
                "ram": ram_name,
                "storage": storage_name,
                "monitor": monitor_name,
                "total_price": total_price
            }

            # Calculate configuration score
            score = self.calculate_configuration_score(config, task)

            # Store configuration
            self.declare(Configuration(
                components=config,
                score=score
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

    @Rule(AS.config << Configuration())
    def select_best_configuration(self, config):
        config_data = {
            "score": config["score"],
            "components": config["components"]
        }
        
        if (self.best_config is None or 
            config_data["score"] > self.best_config["score"]):  # Consistent dictionary access
            self.best_config = config_data
        self.alternative_configs.append(config_data)
    # @Rule(AS.config << Configuration(score=MATCH.score))
    # def select_best_configuration(self, config, score):
    #     if (self.best_config is None or
    #             not hasattr(self.best_config, 'score') or
    #             score > self.best_config.score
    #             ):
    #         self.best_config = config
    #     self.alternative_configs.append(config)


    def format_configuration(self, config):
        if not config:  # If config is None or empty
            return "No configuration available"
        try:
            components = config["components"]

        # Calculate total price by looking up each component's price
            prices = []
            for component_type, component_name in components.items():
                if component_type == 'cpu':
                    price = next(c['price'] for c in self.components['processors'] if c['name'] == component_name)
                elif component_type == 'motherboard':
                    price = next(m['price'] for m in self.components['motherboards'] if m['name'] == component_name)
                elif component_type == 'gpu':
                    price = next(g['price'] for g in self.components['gpus'] if g['name'] == component_name)
                elif component_type == 'ram':
                    price = next(r['price'] for r in self.components['ram'] if r['name'] == component_name)
                elif component_type == 'storage':
                    price = next(s['price'] for s in self.components['storage'] if s['name'] == component_name)
                elif component_type == 'monitor':
                    price = next(m['price'] for m in self.components['monitors'] if m['name'] == component_name)
                prices.append(price)
        
            total_price = sum(prices)
            # total_price = sum(component["price"]
            #                   for component in config["components"].values())
            return f"""
                Configuration Score: {config['score']:.2f}
                Total Price: ${total_price:.2f}
                Components:
                - CPU: {config['components']['cpu']}
                - Motherboard: {config['components']['motherboard']}
                - GPU: {config['components']['gpu']}
                - RAM: {config['components']['ram']}
                - Storage: {config['components']['storage']}
                - Monitor: {config['components']['monitor']}
            """
        except (KeyError, StopIteration):
            return "Invalid configuration data"
        except Exception as e:
            return f"An error occurred: {e}"

    def print_configurations(self):
        print("\n=== BEST CONFIGURATION ===")

        if self.best_config:
            print(self.format_configuration(self.best_config))
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
    engine.declare(UserPreferences(task="gaming", budget=5000))
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
