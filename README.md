# LLMTraveler


📄 **Publication**  
We are delighted to share that our work has been accepted and published in  
*Transportation Research Part C: Emerging Technologies*! 🎉  

Read the paper here:  
[Agentic Large Language Models for Day-to-Day Route Choices](https://www.sciencedirect.com/science/article/pii/S0968090X25003110)  

---------------------------------------------------------------------------------------------------
## Route Choice Modeling Using LLM-based Agent
This project explores the potential of Large Language Models (LLMs) to model human-like route choice behavior by introducing the "LLMTraveler" agent, which integrates an LLM with a memory system for learning and adaptation. The experiments demonstrate that the framework can partially replicate human-like decision-making in route choice, while also providing natural language explanations for its decisions.

---------------------------------------------------------------------------------------------------

## Example visualizations
### Example 1: Individual-level behavior of LLMTravelers

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/gif/route_choices_of_two_agents_small.gif)  
The gif shows the route choice evolution of two LLM-gpt35-based agents (Daniel Nguyen and Jessica Ramirez) in the OW network, traveling between the same origin and destination. Initially, Daniel explored from "Route 0" to "Route 4," while Jessica moved from "Route 4" to "Route 0." Daily fluctuations in travel time influenced their perception of the average travel time for each route, leading to different learning outcomes. After 50 days, Jessica consistently chose "Route 2," while Daniel identified "Route 1" as optimal after 80 days. Their distinct exploration and memory-based decision-making led to different route choices.

The following video shows the route choice evolution of all 30 LLM-gpt35-based agents, including Daniel Nguyen and Jessica Ramirez, traveling from Node 1 to Node 12, which may further aid in understanding their behavior.

https://private-user-images.githubusercontent.com/46119396/398356311-5413ad77-d6ec-4e94-97c3-5560fbd448ae.mp4

---------------------------------------------------------------------------------------------------

### Example 2: Aggregate-level behavior of LLMTravelers

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/gif/LLMTravelers_avg_tt_small.gif)  
The gif shows that all LLMTravelers converge toward a smaller travel time, indicating a move towards user equilibrium, after experiencing initial fluctuations over several days. Although the travel time becomes smaller over time, fluctuations persist, aligning with laboratory experiment findings. Additionally, the performance differences between LLM models of varying sizes are minimal. This suggests that lightweight, open-source LLMs can serve as cost-effective alternatives.

---------------------------------------------------------------------------------------------------

### Example 3: Comparison with MNL and Reinforcement Learning(RL)-based method

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/gif/Diff_methods_avg_tt.gif)  
The gif shows that LLM-gpt35, MNL, and the RL-based agent converge to a similar average travel time of approximately 71.1 minutes. However, both LLM-gpt35 and the RL-based agent exhibit fluctuations around user equilibrium, achieving slightly lower travel times—behavior not observed in the MNL model. The MNL model converges more quickly in the initial days due to its method of sharing experiences, while LLM-gpt35 optimizes choices based on individual experience, which may better reflect real-world behavior. Notably, the RL-based agent requires significantly more data—nearly 4000 days—to match LLM-gpt35’s convergence within 100 days, highlighting the low sample efficiency of RL in this context, and emphasizing the comparatively higher efficiency of LLM-gpt35.

Note: The LLM-gpt35 simulation was run for 100 days, so the average travel time remains unchanged after day 100.

---------------------------------------------------------------------------------------------------

## Setup Instructions

1. **Create and activate a Conda environment**

   ```bash
   conda create -n LLMTraveler python=3.10 -y
   conda activate LLMTraveler
   ```

2. **Install dependencies**
   You can either install packages individually:

   ```bash
   pip install networkx pandas matplotlib openai openpyxl python-dotenv
   ```

   or install all from the lock file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API access with `.env`**

   This project uses environment variables to manage API keys.  
   The recommended way is to use [`python-dotenv`](https://pypi.org/project/python-dotenv/) for local development.

   1. Make sure `python-dotenv` is installed (already included above).  
   2. Create a `.env` file in the project root and add your credentials:

      ```dotenv
      # Example: Azure OpenAI
      AZURE_OPENAI_API_KEY=your_api_key_here
      AZURE_ENDPOINT=https://your-resource-name.openai.azure.com
      AZURE_OPENAI_API_VERSION=2024-07-01-preview

      # (Optional alternative providers)
      # OPENAI_API_KEY=your_api_key_here
      # OPENROUTER_API_KEY=your_api_key_here
      # DEEPSEEK_API_KEY=your_api_key_here
      # OPENAI_BASE_URL=https://openrouter.ai/api/v1
      # OPENAI_BASE_URL=https://api.deepseek.com/v1
      ```

   3. Load environment variables once at startup (already handled in many setups):

      ```python
      from dotenv import load_dotenv
      load_dotenv()  # reads .env into environment variables
      ```

   > **Tip:** Never commit your `.env` file or any API keys to version control.

4. **Run an example**

   ```bash
   python LLMTraveler-OW.py --temperature 0.6
   ```


## Citation

```bibtex
@article{Wang2025LLMTraveler,
  title   = {Agentic Large Language Models for day-to-day route choices},
  author  = {Wang, Leizhen and Duan, Peibo and He, Zhengbing and Lyu, Cheng and Chen, Xin and Zheng, Nan and Yao, Li and Ma, Zhenliang},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {180},
  pages   = {105307},
  year    = {2025},
  doi     = {10.1016/j.trc.2025.105307}
}
