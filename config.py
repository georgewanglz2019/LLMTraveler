import argparse
import json
import os


def save_args(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def get_config(net='ow'):
    # net: 'ow' or 'siouxfalls' or 'oneod'
    parser = argparse.ArgumentParser(description="Example script")

    parser.add_argument('--simulation_days', type=int, default=100, help="Simulation days or iterations")
    
    # Add scenario parameter for OneOD_Net
    if net == 'oneod':
        parser.add_argument('--scenario', type=int, choices=[1, 2, 3, 4, 5], default=5,
                            help="Scenario number for OneOD_Net (1-5)")
    if net == 'siouxfalls':
        parser.add_argument('--travellers_per_agent', type=int, default=600,
                            help="Number of travellers for whom an agent is responsible for making decisions")
    else:  # ow
        parser.add_argument('--travellers_per_agent', type=int, default=20,
                            help="Number of travellers for whom an agent is responsible for making decisions")
    if net == 'siouxfalls':
        parser.add_argument('--init_tt', type=str, choices=["fft", "0", "10", "20", "50", "80", "100"], default="5",
                            help="Initial travel time for unknown routes")
    else:
        parser.add_argument('--init_tt', type=str, choices=["fft", "0", "10", "20", "50", "80", "100"], default="50",
                            help="Initial travel time for unknown routes")
    parser.add_argument('--action_strategy', type=str, choices=["llm", "ucb", "epsilon_greedy", "greedy"],
                        default='llm', help="Route selection strategy")
    parser.add_argument('--llm_model', type=str,
                        choices=["azure_gpt35", "azure_gpt4o", "azure_gpt4omini", "openai_gpt35", "yi-large", "yi-medium", "glm4-air",
                                 "moonshot-v1-8k", "claude-3-5-sonnet-20240620", 'qwen-long', 'qwen-turbo',
                                 'llama-3-70b-instruct', 'llama-3-8b-instruct', 'gemma-2-9b-it', 'llama-3.1-70b-instruct',
                                 'llama-3.1-8b-instruct', 'openrouter_gpt35'],
                        default='azure_gpt35', help="Large language model base")
    if net == 'siouxfalls':
        parser.add_argument('--temperature', type=float, default=0.5,
                            help="Temperature (hyper-parameter) of llm model, which regulates the amount of randomness, \
                            leading to more diverse outputs")
    else:
        parser.add_argument('--temperature', type=float, default=0.3,
                            help="Temperature (hyper-parameter) of llm model, which regulates the amount of randomness, \
                                leading to more diverse outputs")
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help="Epsilon for epsilon greedy strategy, range from (0,1)")
    if net == 'siouxfalls':
        parser.add_argument('--od_demand_file', type=str, default="SiouxFall_half_demand.csv",
                            choices=["OW_trips.csv", "OW_trips2.csv", "SiouxFall_half_demand.csv",
                                     "SiouxFall_full_demand.csv"],
                            help="Od demand matrix file")
    else:
        parser.add_argument('--od_demand_file', type=str, default="OW_trips.csv",
                            choices=["OW_trips.csv", "OW_trips2.csv", "SiouxFall_half_demand.csv",
                                     "SiouxFall_full_demand.csv"],
                            help="Od demand matrix file")
    parser.add_argument('--k', type=int, default=5, help="k shortest path (k)")

    parser.add_argument('--cot', type=str, default="zero-shot-cot", choices=["none", "zero-shot-cot", "manual-cot"],
                        help="Chain-of-thought (CoT) prompt type for LLM")

    parser.add_argument('--personality_distribution', choices=["none", "same", "random"], default="none",
                        help="Set the personality distribution for LLM agents, none means no personality")

    parser.add_argument('--extroversion', choices=["extroverted", "introverted"], default="extroverted",
                        help="Extroversion type")
    parser.add_argument('--agreeableness', choices=["agreeable", "antagonistic"], default="agreeable",
                        help="Agreeableness type")
    parser.add_argument('--conscientiousness', choices=["conscientious", "unconscientious"], default="conscientious",
                        help="Conscientiousness type")
    parser.add_argument('--neuroticism', choices=["neurotic", "emotionally stable"], default="neurotic",
                        help="Neuroticism type")
    parser.add_argument('--openness', choices=["open to experience", "closed to experience"],
                        default="open to experience", help="Openness to experience type")

    parser.add_argument('--is_selfish', type=str, default='True', choices=["True", "False"],
                        help="Whether the agent is selfish")  # use string because in commandline all args are str, not bool

    parser.add_argument('--alpha', type=float, default=0.1, help="Alpha for softmax in day-to-day assignment algorithm")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_config()
    #args.llm_model = 'sss'
    save_args(args, 'default_args.json')


