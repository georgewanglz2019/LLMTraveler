import numpy as np
import time
import random
import os
import json

from LLM_Client import LLM_Client
from load_OW_net_config import *
from utils import create_output_directory, setup_logging, save_dict_to_json, get_random_character
from config import get_config, save_args, load_args

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment variables


class OneOD_Net:
    """
    One Origin-Destination Network simulation with LLM-based route choice.
    Simulates traffic flow between two points with multiple route options.
    """
    def __init__(self, args, scenario: int):
        # scenario: 1, 2, 3, 4, 5 - different network configurations
        self.args = args  # simulation parameters (llm model, temperature, etc.)
        self.scenario_idx = scenario - 1
        self.total_agents = 16  # fixed number of agents for one OD pair
        self.travellers_list = []
        self.every_day_route_choices = {}  # daily route choices: {day: [routes]}
        self.no_day = 0
        self.id = 0
        self.simulation_days = args.simulation_days
        self.num_agents_for_od_dict = {'A_B': 16}  # OD pair: number of agents

        # log and data saving
        self.save_dir = create_output_directory(base_dir='output_one_od', prefix=f'scenario{self.scenario_idx+1}_')
        self.logging = setup_logging(os.path.join(self.save_dir, 'logging.log'))
        save_args(args, os.path.join(self.save_dir, 'args.json'))

        # LLM client
        self.llm_client = LLM_Client(logging=self.logging, temperature=args.temperature, model=args.llm_model,
                                     max_tokens=4096)

        # Network performance parameters for different scenarios
        self.bonus_tt_base_list = [40, 60, 40, 60, 40]  # base travel time for bonus calculation
        self.bonus_tt_base = self.bonus_tt_base_list[self.scenario_idx]
        args.bonus_tt_base = self.bonus_tt_base
        self.performance_func_dict = {
            # Free flow travel times for each scenario: (route_a, route_b)
            'free_flow_tt': [(6, 6), (10, 24), (5, 12), (12, 24), (6, 12)],
            # Travel time increase per vehicle for each scenario: (route_a, route_b)
            'incre_per_veh': [(2, 2), (4, 6), (2, 3), (4, 6), (2, 3)],
        }
        self.route_a_fft = self.performance_func_dict['free_flow_tt'][self.scenario_idx][0]
        self.route_b_fft = self.performance_func_dict['free_flow_tt'][self.scenario_idx][1]
        self.route_a_incre_per_veh = self.performance_func_dict['incre_per_veh'][self.scenario_idx][0]
        self.route_b_incre_per_veh = self.performance_func_dict['incre_per_veh'][self.scenario_idx][1]

        # Initialize route information for current scenario
        self.route_info = {
            'route_name_list': ['route_a', 'route_b'],
            'free_flow_tt': self.performance_func_dict['free_flow_tt'][self.scenario_idx],
            'incre_per_veh': self.performance_func_dict['incre_per_veh'][self.scenario_idx],
        }
        self.names_list = pd.read_csv('Random_Names_200.csv').Names.values.tolist()
        self.persona_list = self.generate_personality_configurations(distribution_type=args.personality_distribution)
        self.agent_info_dict = {}
        for od, num_agent in self.num_agents_for_od_dict.items():
            for _ in range(num_agent):
                persona = self.persona_list.pop(0)  # personality traits for LLM agent
                character = get_random_character()  # demographic characteristics for LLM agent
                name = random.choice(self.names_list)
                traveller = Traveller(llm_client=self.llm_client, unique_id=self.id, net_model=self, name=name,
                                      persona=persona, character=character, od=od, route_info=self.route_info, logging=self.logging,
                                      args=args)
                agent_info = {
                    'id': self.id,
                    'name': name,
                    'persona': persona,
                    'route_info': self.route_info,
                }
                self.agent_info_dict[self.id] = agent_info

                self.id += 1
                self.travellers_list.append(traveller)

        save_dict_to_json(data_dict=self.agent_info_dict, filename=os.path.join(self.save_dir, f'agent_info_dict.json'))

    def generate_personality_configurations(self, distribution_type):
        num_agents = self.total_agents
        def generate_random_personality():
            personality_types = {
                "extroversion": ["extroverted", "introverted"],
                "agreeableness": ["agreeable", "antagonistic"],
                "conscientiousness": ["conscientious", "unconscientious"],
                "neuroticism": ["neurotic", "emotionally stable"],
                "openness": ["open to experience", "closed to experience"]
            }

            selected_personality = {
                "extroversion": random.choice(personality_types["extroversion"]),
                "agreeableness": random.choice(personality_types["agreeableness"]),
                "conscientiousness": random.choice(personality_types["conscientiousness"]),
                "neuroticism": random.choice(personality_types["neuroticism"]),
                "openness": random.choice(personality_types["openness"]),
            }
            return selected_personality

        if distribution_type not in ["same", "random", "none"]:
            raise ValueError("Invalid distribution type. Must be 'same', 'random', or 'proportional'.")

        if distribution_type == "same":
            same_personality = {'extroversion': self.args['extroversion'],
                                'agreeableness': self.args['agreeableness'],
                                'conscientiousness': self.args['conscientiousness'],
                                'neuroticism': self.args['neuroticism'],
                                'openness': self.args['openness']}
            return [same_personality] * num_agents

        elif distribution_type == "random":
            return [generate_random_personality() for _ in range(num_agents)]

        elif distribution_type == "none":
            return [None for _ in range(num_agents)]


    def seed(self, seed):
        # seed
        random.seed(seed)
        np.random.seed(seed)


    def save_lists_to_csv(self, day, directory, lists, column_names):
        """
        Save multiple lists to a single CSV file in the specified directory with given column names.

        Args:
            day (int): The current iteration number.
            directory (str): The directory to save the CSV file.
            lists (list of list): The lists to save.
            column_names (list of str): The column names for each list.
        """
        # Combine lists into a DataFrame
        data = {col: lst for col, lst in zip(column_names, lists)}
        df = pd.DataFrame(data)

        # Define the file path
        file_path = os.path.join(directory, f'day_{day}.csv')

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        #print(f"Saved {file_path}")

    def append_to_file(self, value, filename='output.txt'):
        with open(filename, 'a') as file:
            file.write(f"{value}\n")

    def append_dict_to_json(self, data, file_path):
        if os.path.exists(file_path):
            # If file exists, read existing data and append new data
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
            existing_data.append(data)
        else:
            # If file doesn't exist, create new file with data
            existing_data = [data]

        # Write data to file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4, default=str)

    def simulate_day_to_day(self):
        """
        Main simulation loop: day-to-day route choice with LLM agents.
        Each day, agents choose routes based on historical experience and LLM reasoning.
        """
        relative_gap_list = []
        traveller_unique_id_list = [tra.unique_id for tra in self.travellers_list]
        traveller_name_list = [tra.name for tra in self.travellers_list]
        traveller_od_list = [tra.od for tra in self.travellers_list]

        for day in range(self.simulation_days):
            self.logging.info(f'**************************** Day={day} ****************************')
            day_routes = []
            day_reasons = []
            day_prompt = []
            is_shortest_list = []

            # Each agent chooses route using LLM strategy
            for traveller in self.travellers_list:
                n_try = 0
                self.logging.info(f'---------- traveller_{traveller.unique_id}, name={traveller.name}')
                n_max_try = 5
                while n_try < n_max_try:  # retry mechanism for LLM failures
                    try:
                        self.logging.debug(f'{n_try} times try using llm to get response !')
                        route_choice, reason, prompt, is_shortest = traveller.choose_route()
                        self.logging.info(f'route_choice={route_choice}')
                        self.logging.info(f'reason={reason}')
                        n_try = n_max_try
                    except:
                        n_try += 1
                        self.logging.error(f'something wrong while trying to get response using llm !')
                        pass
                traveller.day += 1

                day_routes.append(route_choice)
                day_reasons.append(reason)
                day_prompt.append(prompt)
                is_shortest_list.append(is_shortest)

            # Network loading: calculate travel times based on route choices
            print(day_routes)
            route_a_count = day_routes.count('route_a')
            route_a_tt = self.route_a_fft + self.route_a_incre_per_veh * route_a_count
            route_b_count = day_routes.count('route_b')
            route_b_tt = self.route_b_fft + self.route_b_incre_per_veh * route_b_count
            assert (route_a_count + route_b_count) == self.total_agents
            day_route_tt = {'route_a': route_a_tt, 'route_b': route_b_tt}
            day_route_tt_list = [day_route_tt[route_choice] for route_choice in day_routes]

            # Calculate relative gap (efficiency measure)
            TSTT = route_a_count * route_a_tt + route_b_count * route_b_tt
            relative_gap = round(TSTT / (min(route_a_tt, route_b_tt) * self.total_agents) - 1, 2)

            relative_gap_list.append(relative_gap)
            self.logging.info(f'relative_gap={relative_gap}')
            self.logging.info(f'relative_gap_list={relative_gap_list}')

            # Update each agent's experience with current day's travel times
            for i, traveller in enumerate(self.travellers_list):
                traveller.add_experience(day=day, route=day_routes[i], tt=[route_a_tt, route_b_tt], reason=day_reasons[i])

            # save data
            self.save_lists_to_csv(day=day, directory=self.save_dir,
                                   lists=[traveller_unique_id_list, traveller_name_list, traveller_od_list, day_routes,
                                          day_reasons, day_route_tt_list, day_prompt, is_shortest_list],
                                   column_names=['unique_id', 'name', 'od', 'route_choice', 'reason', 'route_tt',
                                                 'prompt', 'is_shortest'])
            self.append_to_file(relative_gap, filename=os.path.join(self.save_dir, 'relative_gap.txt'))
            self.append_to_file(TSTT, filename=os.path.join(self.save_dir, 'TSTT.txt'))


        # save traveller tt memory data
        for n, traveller in enumerate(self.travellers_list):  # each traveller/agent
            traveller.tt_memory_dict['name'] = traveller.name
            traveller.tt_memory_dict['od'] = traveller.od
            traveller.tt_memory_dict['unique_id'] = traveller.unique_id
            self.append_dict_to_json(data=traveller.tt_memory_dict,
                                     file_path=os.path.join(self.save_dir, f'tt_memory_{traveller.unique_id}.json'))

        # save cost info
        cost_info_dict = self.llm_client.calculate_cost()
        save_dict_to_json(cost_info_dict, os.path.join(self.save_dir, 'cost_info.json'))


class Traveller():
    """
    Individual agent that makes route choices using LLM-based decision making.
    Each agent has personality traits, demographic characteristics, and travel experience memory.
    """
    def __init__(self, llm_client, unique_id, net_model, name, persona, character, od, route_info, logging, args):
        self.llm_client = llm_client
        self.unique_id = unique_id
        self.od = od  # origin-destination pair
        self.persona = persona  # personality traits for LLM prompting
        self.character = character  # demographic characteristics for LLM prompting
        self.net_model = net_model  # reference to network model
        self.name = name
        self.route_info = route_info
        self.logging = logging
        self.args = args

        self.route_name_list = self.route_info['route_name_list']
        self.route_memory = []
        self.reason_memory = []
        self.last_route_choice = None
        self.experience_memory = {}

        self.action_strategy = args.action_strategy
        self.llm_model = args.llm_model
        self.temperature = args.temperature
        self.epsilon = args.epsilon
        self.cot = args.cot

        self.tt_memory_dict = {'route_a': [20], 'route_b': [20]}  # travel time memory
        self.day = 0
        self.yesterday_route_a_tt = 0
        self.yesterday_route_b_tt = 0
        self.yesterday_tt_dict = {}
        self.bonus_tt_base = args.bonus_tt_base
        self.cum_bonus = 0  # cumulative bonus in RMB

    def add_experience(self, day, route, tt, reason):
        #self.day = day
        self.experience_memory[day] = {'route': route}
        self.route_memory.append(route)
        tt_choice = tt[0] if route == 'route_a' else tt[1]
        self.tt_memory_dict[route].append(tt_choice)
        # self.tt_memory_dict['route_a'].append(tt[0])
        # self.tt_memory_dict['route_b'].append(tt[1])
        self.yesterday_route_a_tt = tt[0]
        self.yesterday_route_b_tt = tt[1]
        self.yesterday_tt_dict = {'route_a': tt[0], 'route_b': tt[1]}
        self.reason_memory.append(reason)
        self.last_route_choice = route

    def choose_route(self):
        strategy = self.action_strategy
        if strategy == 'llm':
            route_choice, reason, question_prompt, is_shortest = self.llm_strategy()
            return route_choice, reason, question_prompt, is_shortest

        # elif strategy == 'ucb':
        #     route_choice, reason, question_prompt, is_shortest = self.ucb_strategy()  # here: reason include 'explore' or 'exploitation'
        #     return route_choice, reason, question_prompt, is_shortest
        #
        # elif strategy == 'epsilon_greedy':
        #     route_choice, reason, question_prompt, is_shortest = self.epsilon_greedy_strategy(epsilon=self.epsilon)   # here: reason include 'explore' or 'exploitation'
        #     return route_choice, reason, question_prompt, is_shortest
        #
        # elif strategy == 'greedy':
        #     route_choice, reason, question_prompt, is_shortest = self.greedy_strategy()   # here: reason include 'explore' or 'exploitation'
        #     return route_choice, reason, question_prompt, is_shortest

        else:
            raise ValueError(strategy + ' (choose route strategy) not exist !!!!!!!!!!!!!')


    def get_query_prompt(self):
        ewma_tt_list = [0, 0]
        if self.day == 0:
            history_trips_str = "Today is the first day, so you have no travel experiences yet. Please make a random selection.\n"
            # history ewmatt
            ewmatt_route_a = round(self.calculate_ewma_tt(self.tt_memory_dict['route_a']), 2)
            route_a_chosen_time = len(self.tt_memory_dict['route_a']) - 1
            ewmatt_route_b = round(self.calculate_ewma_tt(self.tt_memory_dict['route_b']), 2)
            route_b_chosen_time = len(self.tt_memory_dict['route_b']) - 1
            ewma_tt_list = [ewmatt_route_a, ewmatt_route_b]
            history_trips_str += f"Your historical travel experiences for each route over the past {self.day + 1} days are as follows:\n"
            history_trips_str += f"route_a: Chosen {route_a_chosen_time} times, with an Experience Weighted Moving Average Travel Time of {ewmatt_route_a}\n"
            history_trips_str += f"route_b: Chosen {route_b_chosen_time} times, with an Experience Weighted Moving Average Travel Time of {ewmatt_route_b}\n"

        else:
            # yesterday tt and bonus
            history_trips_str = "Here is your historical travel data:\n"
            history_trips_str += (f"Yesterday: route_a’s travel time was {self.yesterday_route_a_tt}, route_b’s travel "
                                  f"time was {self.yesterday_route_b_tt}, and you chose {self.last_route_choice}.\n")
            chose_route_tt = self.yesterday_tt_dict[self.last_route_choice]
            bonus = (self.bonus_tt_base - chose_route_tt)/50
            self.cum_bonus += bonus
            history_trips_str += (f"Yesterday, you received a {bonus} RMB bonus, bringing your cumulative bonus to "
                                  f"{self.cum_bonus} RMB.\n")

            # history ewmatt
            ewmatt_route_a = round(self.calculate_ewma_tt(self.tt_memory_dict['route_a']), 2)
            route_a_chosen_time = len(self.tt_memory_dict['route_a']) - 1
            ewmatt_route_b = round(self.calculate_ewma_tt(self.tt_memory_dict['route_b']), 2)
            route_b_chosen_time = len(self.tt_memory_dict['route_b']) - 1
            ewma_tt_list = [ewmatt_route_a, ewmatt_route_b]
            history_trips_str += f"Your historical travel experiences for each route over the past {self.day+1} days are as follows:\n"
            history_trips_str += f"route_a: Chosen {route_a_chosen_time} times, with an Experience Weighted Moving Average Travel Time of {ewmatt_route_a}\n"
            history_trips_str += f"route_b: Chosen {route_b_chosen_time} times, with an Experience Weighted Moving Average Travel Time of {ewmatt_route_b}\n"

        #selfish_prompt = "You are selfish." if self.args.is_selfish == "True" else ""
        if self.persona == None:
            personality_prompt = ""
        else:
            personality_prompt = (f"You are a character who is {self.persona['extroversion']}, "
                                  f"{self.persona['agreeableness']}, {self.persona['conscientiousness']}, "
                                  f"{self.persona['neuroticism']}, and {self.persona['openness']}.")
        if self.cot == 'none':
            cot_prompt = ""
        elif self.cot == 'zero-shot-cot':
            cot_prompt = f"Please think step by step."
        else: # no manual-cot in one_od net
            raise ValueError(f"Not valid cot strategy: {self.cot}")

        query_prompt = f"""
Your name is {self.name}. {self.character} {personality_prompt} 

You are participating in a congestion game over 100 days (from day 0 to day 99), and today is day {self.day}.\
In this game, your rewards depend on others’ choices. Every day, you need to drive from location A to location B, \
choosing between two routes. At the end of the game, you will be paid a reward consisting of a 15 RMB show-up fee plus \
a performance-based bonus. The bonus is calculated by subtracting your travel time from a fixed reward.

{history_trips_str}

{cot_prompt} Please note that the shorter the travel time for the route you choose each day, the higher the bonus you \
will receive. Keep in mind that this is a game where your travel time is influenced by the choices of others. \
Since others have access to similar information, you must maximize your own benefit while anticipating their decisions. \
Additionally, your selection strategy should include an element of randomness.

Please reason and choose route_a or route_b for today. Respond directly in JSON format, including 'Reason' (e.g., ...) \
and 'Choice' (e.g., route_i), without any additional text or explanations.
        """

        return query_prompt, ewma_tt_list

    def llm_strategy(self):
        query_prompt, ewma_tt_list = self.get_query_prompt()
        messages = [
            # {'role': 'system', 'content': persona_prompt},
            {'role': 'user', 'content': query_prompt}
        ]
        # print('question_prompt=', question_prompt)
        self.logging.info(f'query_prompt={query_prompt}')
        # messages = [{'role':'system', 'content':question_prompt}]
        try:
            json_output = self.llm_client.get_chat_completion_with_retries(messages=messages, max_retries=5)
        except Exception as e:
            # print(f"{e}\nProgram paused . Retrying after 60 s ...")
            self.logging.error(f'Get response from llm failed. Retrying after 60 s ...')
            time.sleep(60)
            json_output = self.llm_client.get_chat_completion_with_retries(messages=messages, max_retries=5)

        route_choice, reason = json_output["Choice"], json_output["Reason"]
        #action = int(route_choice.split('_')[-1])
        action_idx = 0 if route_choice[-1] == 'a' else 1
        # sometimes the llm model may choose route that not in predefined route set, then choose the biggest one
        if action_idx > len(ewma_tt_list) - 1:
            action = len(ewma_tt_list) - 1
            self.logging.error(f'LLM model choose {route_choice} that not in predefined route set, then choose route_{action}')

        action_str = 'a' if action_idx == 0 else 'b'
        route_choice = 'route_' + action_str

        #is_shortest = int(np.argmin(ewma_tt_list)) == action
        is_shortest = ewma_tt_list[action_idx] == min(ewma_tt_list)

        return route_choice, reason, query_prompt, is_shortest


    def calculate_ewma_tt(self, history, alpha=0.2):
        """
        Calculate Experience Weighted Moving Average (EWMA) travel time.

        Args:
            history (list of float): Observed travel time history data.
            alpha (float): Weight decay coefficient.

        Returns:
            float: EWMA travel time based on historical data.
        """
        estimated_time = 0
        if len(history) > 0:
            estimated_time = history[0]  # Use first element as initial estimate
            if len(history) > 1:
                for time in history[1:]:
                    estimated_time = alpha * time + (1 - alpha) * estimated_time
        return estimated_time

def main():
    args = get_config(net='oneod')
    # args.llm_model = "azure_gpt35"
    # args.personality_distribution = 'random'
    # args.temperature = 1.5

    model = OneOD_Net(args=args, scenario=args.scenario)  # scenario=1, 2, 3, 4, 5
    model.simulate_day_to_day()


if __name__ == '__main__':

    main()



