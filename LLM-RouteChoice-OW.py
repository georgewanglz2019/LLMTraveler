import numpy as np
import time
import random
import os
import json

from LLM_Client import LLM_Client
from load_OW_net_config import *
from utils import create_output_directory, setup_logging, get_random_character
from config import get_config, save_args, load_args
from utils import save_dict_to_json


class OW_net:
    """
    Multiple Origin-Destination (OD) Pairs Network simulation with LLM-based route choice.
    Simulates traffic flow across multiple OD pairs with complex network topology.
    """
    def __init__(self, args, env_args):
        self.args = args
        self.net_df = env_args.net_df  # network topology data
        self.demand_df = env_args.demand_df  # OD demand matrix
        self.routes_dict = env_args.all_od_paths  # k-shortest paths for each OD pair
        self.route_info_dict, self.route_id_dict = self.get_route_info(self.routes_dict, self.net_df)
        self.full_demand_df = cp.deepcopy(self.demand_df)  # initial demand

        self.travellers_per_agent = args.travellers_per_agent  # travelers per LLM agent
        self.num_agents_for_od_dict = {k: int(v / self.travellers_per_agent) for k, v in
                                       zip(self.demand_df['od'], self.demand_df['demand'])}  # agents per OD pair
        self.total_agents = np.sum(list(self.num_agents_for_od_dict.values()))
        self.travellers_list = []
        self.every_day_route_choices = {}  # daily route choices: {day: [routes]}
        self.no_day = 0
        self.id = 0
        self.simulation_days = args.simulation_days

        # log and data saving
        self.save_dir = create_output_directory(base_dir='output_ow')
        self.logging = setup_logging(os.path.join(self.save_dir, 'logging.log'))
        save_args(args, os.path.join(self.save_dir, 'args.json'))

        # LLM client
        self.llm_client = LLM_Client(logging=self.logging, temperature=args.temperature, model=args.llm_model,
                                     max_tokens=4096)

        # Generate LLM agents with personality traits and characteristics
        self.names_list = pd.read_csv('Random_Names_200.csv').Names.values.tolist()
        self.persona_list = self.generate_personality_configurations(distribution_type=args.personality_distribution)
        self.agent_info_dict = {}
        for od, num_agent in self.num_agents_for_od_dict.items():
            for _ in range(num_agent):
                persona = self.persona_list.pop(0)  # personality traits for LLM prompting
                name = random.choice(self.names_list)
                character = get_random_character(purpose=False)  # demographic characteristics for LLM prompting
                traveller = Traveller(llm_client=self.llm_client, unique_id=self.id, net_model=self, name=name,
                                      persona=persona, character=character, od=od, route_info=self.route_info_dict[od], logging=self.logging,
                                      args=args)
                agent_info = {
                    'id': self.id,
                    'od': od,
                    'name': name,
                    'persona': persona,
                    'character':character,
                    'route_info': self.route_info_dict[od],
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

    def get_route_info(self, routes_dict, net_df):
        """
        Process route information for each OD pair.
        Creates route embeddings and calculates free flow travel times.
        """
        route_info_dict = {}
        route_id_dict = {}  # {0:[1, 2, 9, 12], 1:[1, 4,8,12],...}
        id = 0
        for od, routes in routes_dict.items():
            route_info_dict[od] = {}

            route_info_dict[od]['route_name_list'] = []
            route_info_dict[od]['route_id_list'] = []
            route_info_dict[od]['route_list'] = []
            route_info_dict[od]['edge_list'] = []
            route_info_dict[od]['route_embedding'] = []  # binary vector: 1 if edge is used, 0 otherwise
            route_info_dict[od]['free_flow_tt'] = []
            route_info_dict[od]['n_edge'] = []
            for j, route in enumerate(routes):
                route_id_dict[id] = route

                route_info_dict[od]['route_name_list'].append('route_'+str(j))
                route_info_dict[od]['route_id_list'].append(id)
                route_info_dict[od]['route_list'].append(route)
                route_edges = self.edges_from_path(route)
                route_info_dict[od]['edge_list'].append(route_edges)
                route_embeddings = net_df.od.isin(route_edges).values.astype(int)
                route_info_dict[od]['route_embedding'].append(route_embeddings)
                route_info_dict[od]['free_flow_tt'].append(np.dot(route_embeddings, net_df.free_flow_time))
                route_info_dict[od]['n_edge'].append(len(route))
                id += 1
        return route_info_dict, route_id_dict


    def update_edge_state(self):
        # Update edge travel times based on current flow
        self.net_df = self.update_net_df(net_df=self.net_df)

    def update_net_df(self, net_df):
        edge_flow = net_df['edge_flow'].values
        free_flow_time = net_df['free_flow_time'].values
        net_df['edge_tt'] = free_flow_time + 0.02 * edge_flow
        return net_df

    def edges_from_path(self, s_path):
        # Convert node sequence to edge list: [1, 3, 4, 5, 6, 2] -> ['1_3', '3_4', '4_5', '5_6', '6_2']
        edges = []
        for i in range(len(s_path) - 1):
            o = str(s_path[i])
            d = str(s_path[i + 1])
            edges.append(o + '_' + d)
        return edges

    def get_total_tt(self):
        # Calculate total system travel time
        total_tt = np.dot(self.net_df.edge_flow.values, self.net_df.edge_tt.values)
        return total_tt

    def seed(self, seed):
        # seed
        random.seed(seed)
        np.random.seed(seed)

    def network_loading(self, route_embeddings, net_df):
        # Assign flow to edges based on route choices
        net_df['edge_flow'] = 0
        for r_embb in route_embeddings:
            net_df['edge_flow'] += r_embb * self.travellers_per_agent

        net_df = self.update_net_df(net_df)

        # Calculate route travel times
        edge_tt = net_df['edge_tt'].values
        route_tt_list = []
        for r_embb in route_embeddings:
            route_tt_list.append(np.dot(r_embb, edge_tt))

        return net_df, route_tt_list

    def get_odpairs_shortest_path(self, net_df, od_pairs, weight, get_sp_cost=True):
        """
        Calculate shortest paths for given OD pairs.
        """
        nx_G = nx.from_pandas_edgelist(net_df, 'init_node', 'term_node', [weight],
                                             create_using=nx.DiGraph())
        sp_dict = {}
        sp_cost_dict = {}
        for od_pair in od_pairs:
            origin, destination = int(od_pair.split('_')[0]), int(od_pair.split('_')[1])

            # Calculate shortest path
            sp = nx.shortest_path(nx_G, source=origin, target=destination, weight=weight)
            sp_dict[od_pair] = sp
            if get_sp_cost:  # Calculate shortest path cost
                sp_length = nx.shortest_path_length(nx_G, source=origin, target=destination, weight=weight)
                sp_cost_dict[od_pair] = sp_length

        return sp_dict, sp_cost_dict

    def get_relative_gap(self, net_df, num_agents_for_od_dict):
        """
        Calculate relative gap (efficiency measure) between current and optimal assignment.
        """
        # Calculate relative gap for each OD pair
        all_od_pairs = list(num_agents_for_od_dict.keys())
        sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=all_od_pairs,
                                                               weight='edge_tt')

        SP_COST = 0  # shortest path cost
        TS_COST = np.dot(net_df['edge_tt'], net_df['edge_flow'])  # current total assignment cost

        edge_tt = net_df['edge_tt'].values

        # for agent_id, action in enumerate(actions):
        for od in all_od_pairs:
            #ass_info = all_assign_info[od]
            num_agents = num_agents_for_od_dict[od]
            SP_COST += sp_cost_dict[od] * num_agents * self.travellers_per_agent

        global_gap = (TS_COST / SP_COST) - 1

        return global_gap, TS_COST

    def get_local_relative_gap(self, net_df, traveller_od_list, day_route_tt_list):
        """
        get local relative gap for each od
        refer to: https://sboyles.github.io/blubook.html
                  https://sboyles.github.io/teaching/ce392c/9-pathbased.pdf

        :param net_df: (pd.DataFrame) network data
        :param traveller_od_list: (list) list of od. e.g., ['1_12', '1_12', ...]
        :param day_route_tt_list: (list) list of travel times. e.g., [12, 25, ...]

        :return local_gaps_dic: (dict)
        """
        ##### get relative gap for each od/agent
        unique_od_list = []
        for od in traveller_od_list:
            if od not in unique_od_list:
                unique_od_list.append(od)
        print('')
        sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=unique_od_list,
                                                               weight='edge_tt')  # {'1_2':[1,2],.}

        local_relative_gap_dict = {}  # e.g., {'1_12': 0.5,...}
        for od in unique_od_list:
            local_TotalTT = 0
            num_agents = 0
            for i, tra_od in enumerate(traveller_od_list):
                if tra_od == od:
                    local_TotalTT += day_route_tt_list[i] * self.travellers_per_agent
                    num_agents += 1
            #num_agents = num_agents_for_od_dict[od]
            local_ShortestTT = sp_cost_dict[od] * num_agents * self.travellers_per_agent
            local_relative_gap_dict[od] = (local_TotalTT / local_ShortestTT) - 1

        return local_relative_gap_dict

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

    def get_day_to_day_assignment(self, alpha=0.05, lamd=0.2):
        alpha = self.args.alpha
        net_df = cp.deepcopy(self.net_df)
        od_demand_dict = {k: int(v) for k, v in
                               zip(self.full_demand_df['od'], self.full_demand_df['demand'])}  # {'2_1':100, ....}

        n_route_dict = {}
        route_allocation_dict = {}  # init by uniform allocation
        route_cost_dict = {}  # traveller believed travel time (cost)
        for od, r_info in self.route_info_dict.items():
            n_route = len(r_info['route_embedding'])
            n_route_dict[od] = n_route
            route_allocation_dict[od] = np.ones(n_route) / n_route
            route_cost_dict[od] = np.ones(n_route) * 50  # init travel time = 50

        # save init data
        self.append_dict_to_json(data=route_allocation_dict,
                                 file_path=os.path.join(self.save_dir, f'route_allocation_-1.json'))
        self.append_dict_to_json(data=route_cost_dict,
                                 file_path=os.path.join(self.save_dir, f'user_believed_route_tt_-1.json'))

        relative_gap_list = []
        for day in range(self.simulation_days):
            self.logging.info(f'**************************** Day={day} ****************************')

            # update network flow
            net_df['edge_flow'] = 0
            for od, demand in od_demand_dict.items():
                route_embeddings = self.route_info_dict[od]['route_embedding']
                for i, r_embb in enumerate(route_embeddings):
                    net_df['edge_flow'] += r_embb * route_allocation_dict[od][i] * demand

            # network loading
            net_df = self.update_net_df(net_df)

            # get new real travel time dict for all od and routes
            edge_tt = net_df['edge_tt'].values
            real_route_tt_dict = {}
            for od, demand in od_demand_dict.items():
                route_embeddings = self.route_info_dict[od]['route_embedding']
                real_route_tt_dict[od] = np.zeros(len(route_embeddings))
                for i, r_embb in enumerate(route_embeddings):
                    real_route_tt_dict[od][i] = np.dot(r_embb, edge_tt)

            # get relative gap
            all_od_pairs = list(od_demand_dict.keys())
            sp_dict, sp_cost_dict = self.get_odpairs_shortest_path(net_df=net_df, od_pairs=all_od_pairs,
                                                                   weight='edge_tt')  # {'1_2':[1,2],.}
            SP_COST = 0  # shortest path cost
            TS_COST = np.dot(net_df['edge_tt'], net_df['edge_flow'])  # current total assignment cost
            for od in all_od_pairs:
                SP_COST += sp_cost_dict[od] * od_demand_dict[od]
            relative_gap = (TS_COST / SP_COST) - 1
            relative_gap_list.append(relative_gap)

            self.logging.info(f'relative_gap={relative_gap}')
            self.logging.info(f'relative_gap_list={relative_gap_list}')
            self.append_to_file(relative_gap, filename=os.path.join(self.save_dir, 'relative_gap.txt'))
            self.append_to_file(TS_COST, filename=os.path.join(self.save_dir, 'TSTT.txt'))

            # update traveller believed travel time (cost)
            for od, cost in route_cost_dict.items():
                route_cost_dict[od] = (1 - lamd) * cost + lamd * real_route_tt_dict[od]

            # new assignment allocation
            def update_path_probabilities(travel_times, alpha):
                # 使用softmax更新路径选择概率
                exp_neg_t = np.exp(-alpha * travel_times)
                probabilities = exp_neg_t / np.sum(exp_neg_t)
                return probabilities
            for od, allocation in route_allocation_dict.items():
                route_allocation_dict[od] = update_path_probabilities(route_cost_dict[od], alpha)

            self.logging.info(f'real_route_tt_dict={real_route_tt_dict}')
            self.logging.info(f'route_cost_dict={route_cost_dict}')
            self.logging.info(f'route_allocation_dict={route_allocation_dict}')

            self.append_dict_to_json(data=route_allocation_dict,
                                     file_path=os.path.join(self.save_dir, f'route_allocation_{day}.json'))
            self.append_dict_to_json(data=real_route_tt_dict,
                                     file_path=os.path.join(self.save_dir, f'real_route_tt_{day}.json'))
            self.append_dict_to_json(data=route_cost_dict,
                                     file_path=os.path.join(self.save_dir, f'user_believed_route_tt_{day}.json'))

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
            day_route_embeddings = []
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
                        route_embedding = traveller.get_route_embedding(route=route_choice)
                        n_try = n_max_try
                    except:
                        n_try += 1
                        self.logging.error(f'something wrong while trying to get response using llm !')
                        pass

                day_routes.append(route_choice)
                day_reasons.append(reason)
                day_route_embeddings.append(route_embedding)
                day_prompt.append(prompt)
                is_shortest_list.append(is_shortest)

            # Network loading: calculate travel times and efficiency measures
            self.net_df, day_route_tt_list = self.network_loading(route_embeddings=day_route_embeddings, net_df=self.net_df)
            relative_gap, TSTT = self.get_relative_gap(self.net_df, self.num_agents_for_od_dict)
            local_relative_gap_dict = self.get_local_relative_gap(self.net_df, traveller_od_list, day_route_tt_list)
            relative_gap_list.append(relative_gap)
            self.logging.info(f'relative_gap={relative_gap}')
            self.logging.info(f'relative_gap_list={relative_gap_list}')
            self.logging.info(f'local_relative_gap_list={local_relative_gap_dict}')

            # Update each agent's experience with current day's travel times
            for i, traveller in enumerate(self.travellers_list):
                traveller.add_experience(day=day, route=day_routes[i], tt=day_route_tt_list[i], reason=day_reasons[i])
                traveller.update_ewma_tt()

            # save data
            self.save_lists_to_csv(day=day, directory=self.save_dir,
                                   lists=[traveller_unique_id_list, traveller_name_list, traveller_od_list, day_routes,
                                          day_reasons, day_route_tt_list, day_prompt, is_shortest_list],
                                   column_names=['unique_id', 'name', 'od', 'route_choice', 'reason', 'route_tt',
                                                 'prompt', 'is_shortest'])
            self.append_to_file(relative_gap, filename=os.path.join(self.save_dir, 'relative_gap.txt'))
            self.append_to_file(TSTT, filename=os.path.join(self.save_dir, 'TSTT.txt'))

            local_relative_gap_dict['day'] = day
            self.append_dict_to_json(data=local_relative_gap_dict,
                                     file_path=os.path.join(self.save_dir, 'local_gap.json'))

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

        if args.init_tt == 'fft':
            self.tt_memory_dict = {r_n: [fft] for r_n, fft in zip(self.route_name_list, self.route_info['free_flow_tt'])}
        else:
            self.tt_memory_dict = {r_n: [int(args.init_tt)] for r_n, fft in zip(self.route_name_list, self.route_info['free_flow_tt'])}
        self.ewma_tt_dict = {r_n: fft for r_n, fft in zip(self.route_name_list, self.route_info['free_flow_tt'])}

    def add_experience(self, day, route, tt, reason):
        self.experience_memory[day] = {'route': route}
        self.route_memory.append(route)
        self.tt_memory_dict[route].append(tt)
        self.reason_memory.append(reason)
        self.last_route_choice = route

    def get_route_embedding(self, route):
        #print('route:', route)
        index = int(route.split('_')[1])
        route_embedding = self.route_info['route_embedding'][index]
        return route_embedding


    def choose_route(self):
        strategy = self.action_strategy
        if strategy == 'llm':
            route_choice, reason, question_prompt, is_shortest = self.llm_strategy()
            return route_choice, reason, question_prompt, is_shortest

        elif strategy == 'ucb':
            route_choice, reason, question_prompt, is_shortest = self.ucb_strategy()  # here: reason include 'explore' or 'exploitation'
            return route_choice, reason, question_prompt, is_shortest

        elif strategy == 'epsilon_greedy':
            route_choice, reason, question_prompt, is_shortest = self.epsilon_greedy_strategy(epsilon=self.epsilon)   # here: reason include 'explore' or 'exploitation'
            return route_choice, reason, question_prompt, is_shortest

        elif strategy == 'greedy':
            route_choice, reason, question_prompt, is_shortest = self.greedy_strategy()   # here: reason include 'explore' or 'exploitation'
            return route_choice, reason, question_prompt, is_shortest

        else:
            raise ValueError(strategy + ' (choose route strategy) not exist !!!!!!!!!!!!!')


    def get_query_prompt(self):
        history_trips_str = ""
        ewma_tt_list = []
        for r_n, tt_list in self.tt_memory_dict.items():
            chosen_times = len(tt_list)
            #mean_travel_time = np.mean(tt_list)
            #std_travel_time = np.std(tt_list)
            ewma_tt = self.calculate_ewma_tt(tt_list)
            ewma_tt_list.append(ewma_tt)
            # history_trips_str += f"{r_n}:, Chosen {chosen_times} times, Experience Weighted Moving Average Travel time = {ewma_tt}\n"
            history_trips_str += f"{r_n}: Chosen {chosen_times} times, Experience Weighted Moving Average Travel time = {ewma_tt};\n"


        selfish_prompt = "You are selfish." if self.args.is_selfish == "True" else ""
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
        elif self.cot == 'manual-cot':
            cot_prompt = """
--Example 1:
Here is your historical travel data:
route_0:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
route_1:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
route_2:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
route_3:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
route_4:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
Response:
{
  "Reason": "All routes have been chosen once and have equal Experience Weighted Moving Average Travel time of 50. Since all routes are equally optimal based on historical data, any route can be chosen.",
  "Choice": "route_1"
}

--Example 2:
Here is your historical travel data:
route_0:, Chosen 1 times, Experience Weighted Moving Average Travel time = 50
route_1:, Chosen 1 times, Experience Weighted Moving Average Travel time = 0
route_2:, Chosen 1 times, Experience Weighted Moving Average Travel time = 65
route_3:, Chosen 1 times, Experience Weighted Moving Average Travel time = 0
route_4:, Chosen 1 times, Experience Weighted Moving Average Travel time = 0
Response:
{
  "Reason": "Exploring route 1, route 3 and route 4 with an EWMATT of 0 minutes could potentially reveal a faster route. Given the need to explore less-traveled routes to optimize future travel times, route 1 is chosen.",
  "Choice": "route_1"
}

--Example 3:
Here is your historical travel data:
route_0:, Chosen 7 times, Experience Weighted Moving Average Travel time = 70.90
route_1:, Chosen 9 times, Experience Weighted Moving Average Travel time = 72.05
route_2:, Chosen 5 times, Experience Weighted Moving Average Travel time = 73.81
route_3:, Chosen 5 times, Experience Weighted Moving Average Travel time = 72.77
route_4:, Chosen 5 times, Experience Weighted Moving Average Travel time = 70.57
Response:
{
  "Reason": "Route 4 has the lowest Experience Weighted Moving Average Travel time at 70.57.",
  "Choice": "route_4"
}

--Example 4:
Here is your historical travel data:
route_0:, Chosen 18 times, Experience Weighted Moving Average Travel time = 70.38
route_1:, Chosen 11 times, Experience Weighted Moving Average Travel time = 72.00
route_2:, Chosen 2 times, Experience Weighted Moving Average Travel time = 74.68
route_3:, Chosen 9 times, Experience Weighted Moving Average Travel time = 73.611
route_4:, Chosen 7 times, Experience Weighted Moving Average Travel time = 72.04
Response:
{
  "Reason": "Exploring route_2 despite its higher Experience Weighted Moving Average Travel time allows for gathering more data and potentially discovering a new optimal route. The limited data on route_2 warrants further exploration to accurately assess its viability.",
  "Choice": "route_2"
}

--Example 5:
Here is your historical travel data:
route_0:, Chosen 30 times, Experience Weighted Moving Average Travel time = 59.06
route_1:, Chosen 44 times, Experience Weighted Moving Average Travel time = 57.54
route_2:, Chosen 8 times, Experience Weighted Moving Average Travel time = 61.21
route_3:, Chosen 5 times, Experience Weighted Moving Average Travel time = 59.77
route_4:, Chosen 6 times, Experience Weighted Moving Average Travel time = 60.94
Response:
{
  "Reason": "route_1 has the lowest Experience Weighted Moving Average Travel time (57.54) and has been chosen the most frequently (44 times), suggesting it is a reliable and efficient route.",
  "Choice": "route_1"
}
            """
        else:
            raise ValueError('cot must be "none" or "zero-shot-cot" or "manual-cot"')

        query_prompt = f"""
Your name is {self.name}. {selfish_prompt} {self.character} {personality_prompt} 
Every day, you need to drive from location {self.od.split('_')[0]} to location {self.od.split('_')[-1]} to go shopping.
Based on your historical travel experiences, you have to choose one route out of {len(self.tt_memory_dict)} possible routes.
{cot_prompt}
Here is your historical travel data:
{history_trips_str}
Your choice should optimize your travel time by considering both well-traveled routes and those less explored.
Respond directly in JSON format, including 'Reason' (e.g., ...) and 'Choice' (e.g., route_i), without any additional \
text or explanations.
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

        # choice 有可能出现 'route_1' or 'route 1'
        if '_' in route_choice:
            #print('!')
            action = int(route_choice.split('_')[-1])
        else:
            action = int(route_choice.split(' ')[-1])
        # sometimes the llm model may choose route that not in predefined route set, then choose the biggest one
        if action > len(ewma_tt_list) - 1:
            action = len(ewma_tt_list) - 1
            self.logging.error(f'LLM model choose {route_choice} that not in predefined route set, then choose route_{action}')

        route_choice = 'route_' + str(action)

        #is_shortest = int(np.argmin(ewma_tt_list)) == action
        is_shortest = ewma_tt_list[action] == min(ewma_tt_list)

        return route_choice, reason, query_prompt, is_shortest

    def epsilon_greedy_strategy(self, epsilon=0.2):
        def epsilon_greedy(Q, epsilon):
            """
            Epsilon-greedy strategy for action selection.

            Args:
                Q: Action value estimates (list or np.array)
                epsilon: Exploration probability (0 to 1)

            Returns:
                Selected action (int)
            """
            if random.uniform(0, 1) < epsilon:
                # Exploration: randomly select an action
                return random.randint(0, len(Q) - 1), 'explore'
            else:
                # Exploitation: select action with highest value
                return np.argmax(Q), 'exploitation'

        Q_list = []
        for r_n, tt_list in self.tt_memory_dict.items():
            #chosen_times = len(tt_list)
            Q_list.append(- self.calculate_ewma_tt(tt_list))  # note here use negative tt

        action, reason = epsilon_greedy(Q=Q_list, epsilon=epsilon)
        #is_shortest = int(np.argmax(Q_list)) == int(action)
        is_shortest = Q_list[action] == max(Q_list)

        return 'route_' + str(action), reason, '', is_shortest

    def ucb_strategy(self):
        def ucb(Q, N, t):
            """
            Upper Confidence Bound (UCB) strategy for action selection.

            Args:
                Q: Action value estimates (list or np.array)
                N: Number of times each action was selected (list or np.array)
                t: Current time step

            Returns:
                Selected action (int)
            """
            c = 1  # UCB exploration parameter, can be adjusted as needed
            Q = np.array(Q)
            N = np.array(N)
            ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-5))
            return np.argmax(ucb_values)

        Q_list = []
        N_list = []
        t = 0
        for r_n, tt_list in self.tt_memory_dict.items():
            chosen_time = len(tt_list)
            N_list.append(chosen_time)
            t += chosen_time
            Q_list.append(- self.calculate_ewma_tt(tt_list))  # note here use negative tt

        action = ucb(Q=Q_list, N=N_list, t=t)
        #is_shortest = int(np.argmax(Q_list)) == int(action)
        is_shortest = Q_list[action] == max(Q_list)

        return 'route_' + str(action), '', '', is_shortest

    def greedy_strategy(self):
        Q_list = []
        for r_n, tt_list in self.tt_memory_dict.items():
            #chosen_times = len(tt_list)
            Q_list.append(- self.calculate_ewma_tt(tt_list))  # note here use negative tt

        action = np.argmax(Q_list)
        #is_shortest = int(np.argmax(Q_list)) == int(action)
        is_shortest = Q_list[action] == max(Q_list)

        return 'route_' + str(action), '', '', is_shortest

    def update_ewma_tt(self):
        for route_name, tt_list in self.tt_memory_dict.items():
            self.ewma_tt_dict[route_name] = self.calculate_ewma_tt(tt_list)

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
        return round(estimated_time, 1)

def main():
    # args = get_config()
    #
    # cn_llm_models = ["yi-medium"]
    # # args.llm_model = "yi-large"  # "azure_gpt35" "azure_gpt4o", "openai_gpt35", "yi-large", "yi-medium", "glm4-air",
    # #                           # "kimi", "claude-3-5-sonnet-20240620", 'qwen-long'
    # env_args = load_OW_net_config(od_demand_file=args.od_demand_file, k=args.k)
    #
    # for llm_model in cn_llm_models:
    #     args.llm_model = llm_model
    #     model = OW_net(env_args=env_args, args=args)
    #     model.simulate_day_to_day()

    args = get_config()
    #args.is_selfish = "True"
    # args.personality_distribution = 'random'
    env_args = load_OW_net_config(od_demand_file=args.od_demand_file, k=args.k)
    model = OW_net(env_args=env_args, args=args)
    model.simulate_day_to_day()


if __name__ == '__main__':

    main()

    # test get_day_to_day_assignment()
    # args = get_config()
    # args.alpha = 0.3
    #
    #
    # # args.llm_model = "yi-large"  # "azure_gpt35" "azure_gpt4o", "openai_gpt35", "yi-large", "yi-medium", "glm4-air",
    # #                           # "kimi", "claude-3-5-sonnet-20240620", 'qwen-long'
    # env_args = load_OW_net_config(od_demand_file=args.od_demand_file, k=args.k)
    #
    # model = OW_net(env_args=env_args, args=args)
    # model.get_day_to_day_assignment()


