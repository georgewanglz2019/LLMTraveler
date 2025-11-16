from openai import AzureOpenAI, OpenAI
import pandas as pd
import json
import time
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env into environment variables


class LLM_Client:
    def __init__(self, logging, temperature=0., model='azure_gpt35', max_tokens=4096):
        self.logging = logging
        self.temperature = temperature
        self.model = model

        self.max_tokens = max_tokens

        self.total_completion_tokens_usage = 0
        self.total_prompt_tokens_usage = 0

    def get_chat_completion(self, messages):
        model = self.model
        temperature = self.temperature
        max_tokens = self.max_tokens

        if "azure" in model:
            # Azure模型部署名称映射
            azure_model_mapping = {
                "azure_gpt35": "GPT35_LingoTrip",
                "azure_gpt4o": "gpt-4o",
                "azure_gpt4omini": "gpt-4o-mini"
            }
            
            client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-07-01-preview"
            )

            response = client.chat.completions.create(
                model=azure_model_mapping[model],
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content


        elif model == "openai_gpt35":
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # model = "deployment_name".
                temperature=temperature,
                messages=messages,
                max_tokens=max_tokens
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        elif "yi-" in model:
            client = OpenAI(
                api_key=os.getenv("YI_API_KEY"),
                base_url="https://api.lingyiwanwu.com/v1"
            )

            response = client.chat.completions.create(
                model=model,
                temperature=temperature,  # range from 0 to 2
                messages=messages,
                max_tokens=max_tokens
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        elif model == "glm4-air":
            client = OpenAI(
                api_key=os.getenv('ZHIPU_API_KEY'),
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )

            # Zhipu temperature 取值范围是：(0.0, 1.0)，不能等于 0
            temperature /= 2
            temperature = 0.01 if temperature <= 0.01 else temperature
            temperature = 0.99 if temperature >= 0.99 else temperature
            #print(temperature)

            response = client.chat.completions.create(
                #model="glm-4",  # too expensive
                model='glm-4-air',
                messages=messages,
                #top_p=0.7,
                temperature=temperature,  # Zhipu temperature 取值范围是：(0.0, 1.0)，不能等于 0
                max_tokens=max_tokens,
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        elif model == "moonshot-v1-8k":
            client = OpenAI(
                api_key=os.getenv("KIMI_API_KEY"),
                base_url="https://api.moonshot.cn/v1",
            )
            temperature /= 2  # range [0, 1]
            response = client.chat.completions.create(
                model="moonshot-v1-8k",
                temperature=temperature,  # range [0, 1]
                messages=messages,
                max_tokens=max_tokens
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        elif model == "claude-3-5-sonnet-20240620":
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
            temperature /= 2  # range [0, 1]
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",  # https://docs.anthropic.com/en/docs/about-claude/models
                max_tokens=max_tokens,
                temperature=temperature,  # Ranges from 0.0 to 1.0
                messages=messages,
            )

            res_content = response.content[0].text
            completion_tokens_usage = response.usage.output_tokens
            prompt_tokens_usage = response.usage.input_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        elif "qwen-" in model:
            client = OpenAI(
                api_key=os.getenv("ALIBABA_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            max_tokens = 2000 if max_tokens >= 2000 else max_tokens
            temperature = 0.01 if temperature <= 0.01 else temperature
            temperature = 1.99 if temperature >= 1.99 else temperature

            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,  # 取值范围：[0, 2)，不建议取值为0，无意义。
                messages=messages,
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content
        elif model in ['openrouter_gpt35', 'gemma-2-9b-it', 'gpt-o1'] or 'llama-' in model:
            # OpenRouter模型名称映射
            openrouter_model_mapping = {
                'llama-3-70b-instruct': 'meta-llama/llama-3-70b-instruct',
                'llama-3-8b-instruct': 'meta-llama/llama-3-8b-instruct',
                'llama-3.1-70b-instruct': 'meta-llama/llama-3.1-70b-instruct',
                'llama-3.1-8b-instruct': 'meta-llama/llama-3.1-8b-instruct',
                'openrouter_gpt35': 'openai/gpt-3.5-turbo-0125',
                'gemma-2-9b-it': 'google/gemma-2-9b-it',
                'gpt-o1': 'openai/o1-preview-2024-09-12'
            }

            # https://openrouter.ai/docs/quick-start
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv('OPENROUTER_API_KEY'),
            )

            response = client.chat.completions.create(
                model=openrouter_model_mapping[model],
                temperature=temperature,
                messages=messages,
            )

            res_content = response.choices[0].message.content
            completion_tokens_usage = response.usage.completion_tokens
            prompt_tokens_usage = response.usage.prompt_tokens

            self.total_completion_tokens_usage += completion_tokens_usage
            self.total_prompt_tokens_usage += prompt_tokens_usage

            return res_content

        else:
            raise ValueError(f"Model ({model}) not found")


    def load_json_response(self, res_content):
        try:  # 带Markdown code block notation e.g., '\n```json\n{\n  "Choice": "ACDB",\n
              # "Reason": "The travel time for ACDB was the shortest yesterday."\n}\n```'
            json_str = res_content.strip()[7:-3].strip()
            data = json.loads(json_str)
            route, reason = data["Choice"].lower(), data["Reason"]
            return data
        except:
            try:
                data = json.loads(res_content)
                route, reason = data["Choice"].lower(), data["Reason"]
                return data
            except:
                try:
                    json_str = res_content.strip()[3:-3].strip()
                    data = json.loads(json_str)
                    route, reason = data["Choice"].lower(), data["Reason"]
                    return data

                except:
                    print('Not Json response!!!!!!!!!!!!!!!!!!!!!!')
                    raise ValueError('Not Json response!')


    def get_chat_completion_with_retries(self, messages, max_retries=5):
        success = False
        retry = 0
        max_retries = max_retries
        while retry < max_retries and not success:
            try:
                response = self.get_chat_completion(messages=messages)
                json_res = self.load_json_response(response)
                success = True
            except Exception as e:
                #print(f"Error: {e}\ nRetrying ...")
                self.logging.error(f'traveller get_chat_completion from llm failed !')
                retry += 1
                time.sleep(5)

        return json_res  # {'Choice': 'ACDB', 'Reason': 'The travel time for ACDB was the shortest yesterday.'}

    def calculate_cost(self):
        pricing_df = pd.read_csv('LLM_pricing.csv')
        model = self.model
        # 模型版本映射
        model_version_mapping = {
            "azure_gpt35": 'GPT-3.5-Turbo-1106',
            "openai_gpt35": 'GPT-3.5-Turbo-1106',
            "azure_gpt4o": 'GPT-4o',
            "azure_gpt4omini": 'GPT-4o-mini',
            "yi-large": "yi-large",
            "yi-medium": "yi-medium",
            "glm4-air": 'glm-4-air',
            "moonshot-v1-8k": 'moonshot-v1-8k',
            "claude-3-5-sonnet-20240620": 'claude-3-5-sonnet-20240620',
            "qwen-long": "qwen-long",
            "qwen-turbo": "qwen-turbo",
            "llama-3-70b-instruct": 'llama-3-70b-instruct',
            "llama-3-8b-instruct": 'llama-3-8b-instruct',
            "llama-3.1-70b-instruct": 'llama-3.1-70b-instruct',
            "llama-3.1-8b-instruct": 'llama-3.1-8b-instruct',
            "gemma-2-9b-it": 'gemma-2-9b-it',
            "openrouter_gpt35": 'openrouter_gpt35',
            "gpt-o1": 'gpt-o1'
        }
        
        if model in model_version_mapping:
            model_version = model_version_mapping[model]

        else:
            raise ValueError(f'unknown model {model} when calculating cost')

        input_unit_cost = \
        pricing_df.loc[pricing_df.name == model_version, 'input_pricing_per_1Mtokens_Yuan'].values[0]
        output_unit_cost = \
        pricing_df.loc[pricing_df.name == model_version, 'output_pricing_per_1Mtokens_Yuan'].values[0]
        input_cost = self.total_prompt_tokens_usage / 1e6 * input_unit_cost
        completion_cost = self.total_completion_tokens_usage / 1e6 * output_unit_cost
        total_cost = input_cost + completion_cost

        self.logging.info(
            f'total input token={self.total_prompt_tokens_usage}, total completion token={self.total_completion_tokens_usage}')
        self.logging.info(f'total input cost={input_cost}, total completion cost={completion_cost}')
        self.logging.info(f'total cost={total_cost} Yuan')

        cost_info = {'total_prompt_tokens_usage': self.total_prompt_tokens_usage,
                     'total_completion_tokens_usage': self.total_completion_tokens_usage,
                     'input_unit_cost': input_unit_cost, 'output_unit_cost': completion_cost,
                     'input_cost': input_cost, 'completion_cost': completion_cost,
                     'total_cost': total_cost, 'unit': 'Yuan'}

        return cost_info


if __name__ == '__main__':
    from utils import setup_logging
    
    # Define models to test - modify this list as needed
    models_to_test = [
        'azure_gpt35', 'azure_gpt4o', 'azure_gpt4omini',
        'openrouter_gpt35', 'gemma-2-9b-it', 'gpt-o1', 
        'llama-3-70b-instruct', 'llama-3-8b-instruct', 
        'llama-3.1-70b-instruct', 'llama-3.1-8b-instruct'
    ]
    
    logging = setup_logging(os.path.join('LLM_Client_test.log'))
    
    prompt = """

Your name is James Rivera. You are selfish. You are a female character, aged between 55 and 64, with a low income level, an employee, with a high school education, and risk-neutral. You are a character who is introverted, agreeable, conscientious, neurotic, and open to experience. 
Every day, you need to drive from location 1 to location 8 to go shopping.
Based on your historical travel experiences, you have to choose one route out of 5 possible routes.
Please think step by step.
Here is your historical travel data:
route_0: Chosen 2 times, Experience Weighted Moving Average Travel time = 58.6;
route_1: Chosen 1 times, Experience Weighted Moving Average Travel time = 50.0;
route_2: Chosen 1 times, Experience Weighted Moving Average Travel time = 50.0;
route_3: Chosen 1 times, Experience Weighted Moving Average Travel time = 50.0;
route_4: Chosen 1 times, Experience Weighted Moving Average Travel time = 50.0;

Your choice should optimize your travel time by considering both well-traveled routes and those less explored.
Respond directly in JSON format, including 'Reason' (e.g., ...) and 'Choice' (e.g., route_i), without any additional text or explanations.
        
    """

    messages = [
            {'role': 'user', 'content': prompt}
        ]
    
    # Test results storage
    working_models = []
    failed_models = []
    cost_summary = {}
    
    print("=" * 60)
    print("Testing LLM Models...")
    print("=" * 60)
    
    for model in models_to_test:
        print(f"\nTesting: {model}")
        try:
            # Create new client instance
            client = LLM_Client(logging=logging, temperature=0.5, model=model)
            
            # Try to get response
            response = client.get_chat_completion(messages=messages)
            
            # Try to parse JSON response
            j_res = client.load_json_response(response)
            
            print(f"✓ {model} - SUCCESS")
            print(f"  Response: {j_res}")
            
            # Test cost calculation
            try:
                cost_info = client.calculate_cost()
                print(f"  Cost: {cost_info['total_cost']:.4f} {cost_info['unit']}")
                print(f"  Tokens: {cost_info['total_prompt_tokens_usage']} input, {cost_info['total_completion_tokens_usage']} output")
                cost_summary[model] = cost_info
            except Exception as cost_error:
                print(f"  Cost calculation failed: {str(cost_error)}")
                cost_summary[model] = None
            
            working_models.append(model)
            
        except Exception as e:
            print(f"✗ {model} - FAILED: {str(e)}")
            failed_models.append(model)
    
    # Print test results summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    print(f"\n✓ Working Models ({len(working_models)}):")
    for model in working_models:
        print(f"  - {model}")
    
    print(f"\n✗ Failed Models ({len(failed_models)}):")
    for model in failed_models:
        print(f"  - {model}")
    
    print(f"\nTotal: {len(models_to_test)} models, {len(working_models)} working, {len(failed_models)} failed")
    
    # Print cost comparison
    if cost_summary:
        print(f"\n" + "=" * 60)
        print("Cost Comparison:")
        print("=" * 60)
        
        # Sort by cost
        sorted_costs = sorted([(model, info['total_cost']) for model, info in cost_summary.items() if info], 
                             key=lambda x: x[1])
        
        for model, cost in sorted_costs:
            tokens = cost_summary[model]
            print(f"{model:25} | {cost:8.4f} {tokens['unit']:4} | {tokens['total_prompt_tokens_usage']:4} input, {tokens['total_completion_tokens_usage']:4} output")


