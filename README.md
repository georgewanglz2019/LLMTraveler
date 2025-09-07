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

3. **Configure LLM access (choose one provider)**

   > The project uses environment variables by default.  
   > In our code (see `LLM_Client.py`), Azure OpenAI is the default:
   >
   > ```python
   > client = AzureOpenAI(
   >  azure_endpoint=os.getenv("AZURE_ENDPOINT"),
   >  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
   >  api_version="2024-07-01-preview"
   > )
   > ```

   ### Option A — Azure OpenAI (default in repo)

   1) **Get access & deploy a model** via Azure OpenAI.  
   2) **Set environment variables**:

   - **macOS / Linux (bash/zsh):**

     ```bash
     export AZURE_OPENAI_API_KEY="your_api_key_here"
     export AZURE_ENDPOINT="https://your-resource-name.openai.azure.com"
     ```

   - **Windows (PowerShell):**

     ```powershell
     setx AZURE_OPENAI_API_KEY "your_api_key_here"
     setx AZURE_ENDPOINT "https://your-resource-name.openai.azure.com"
     ```

   *(Optional)* If your deployment or SDK requires explicit version or deployment name, also set:

   ```bash
   export AZURE_OPENAI_API_VERSION="2024-07-01-preview"
   export AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
   ```

   ### Option B — OpenAI (platform.openai.com)

   If you choose OpenAI directly, set:

   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```

   and make sure your client code uses the OpenAI endpoint.

   ### Option C — OpenRouter (multi-provider gateway)

   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```

   (You may also need to set a base URL in your client if not using their SDK:
   `OPENAI_BASE_URL="https://openrouter.ai/api/v1"`.)

   ### Option D — DeepSeek

   ```bash
   export DEEPSEEK_API_KEY="your_api_key_here"
   ```

   (If required by your client, also set the base URL:
   `OPENAI_BASE_URL="https://api.deepseek.com/v1"`.)

   ---

   #### Using `python-dotenv` (recommended for local dev)

   1) Ensure `python-dotenv` is installed (already included above).  
   2) Create a `.env` file in the project root:

   ```dotenv
   # Choose one provider; Azure is shown here
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_ENDPOINT=https://your-resource-name.openai.azure.com
   AZURE_OPENAI_API_VERSION=2024-07-01-preview
   # AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   
   # (Alternative providers)
   # OPENAI_API_KEY=your_api_key_here
   # OPENROUTER_API_KEY=your_api_key_here
   # DEEPSEEK_API_KEY=your_api_key_here
   # OPENAI_BASE_URL=https://openrouter.ai/api/v1
   # OPENAI_BASE_URL=https://api.deepseek.com/v1
   ```

   3) Make sure your Python entry loads it once at startup (already done in many setups):

   ```python
   from dotenv import load_dotenv
   load_dotenv()  # reads .env into environment variables
   ```

   > **Security tip:** Never commit your `.env` or API keys to version control.  
   > Add `.env` to `.gitignore`.

4. **Run an example**

   ```bash
   python LLMTraveler-OW.py --temperature 0.6
   ```

   Common flags (examples):

   ```bash
   # adjust randomness
   --temperature 0.4
   
   # select a model/deployment if your code exposes it
   --model gpt-4o-mini
   
   # set a random seed for reproducibility
   --seed 42
   ```

---

## Troubleshooting

- **`KeyError` or authentication errors**  
  Ensure the correct environment variables are set for the provider you chose. Restart your shell (or `source ~/.zshrc` / reopen PowerShell) after `setx` on Windows.

- **Azure-specific: “deployment not found / use deployment name”**  
  Azure uses **deployment names** (not raw model names). Make sure `AZURE_OPENAI_DEPLOYMENT` matches the deployment you created in the Azure portal.

- **`.env` not taking effect**  
  Confirm `python-dotenv` is installed and `load_dotenv()` is executed before reading `os.getenv(...)`.

- **Network / 429 rate limits**  
  Reduce concurrency or temperature, add retry logic, or check account quota.


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
