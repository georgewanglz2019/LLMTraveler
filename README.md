# LLMTraveler

## Route Choice Modeling Using LLM-based Agent
This project explores the potential of Large Language Models (LLMs) to model human-like route choice behavior by introducing the "LLMTraveler" agent, which integrates an LLM with a memory system for learning and adaptation. The study systematically evaluates the LLMTraveler’s ability to replicate human-like decision-making through two stages of day-to-day (DTD) congestion games: (1) analyzing its route-switching behavior in single origin-destination (OD) pair scenarios, where it demonstrates patterns that align with laboratory data but cannot be fully explained by traditional models, and (2) testing its ability to model adaptive learning behaviors in multi-OD scenarios on the Ortuzar and Willumsen (OW) network, producing results comparable to Multinomial Logit (MNL) and Reinforcement Learning (RL) models. These experiments demonstrate that the framework can partially replicate human-like decision-making in route choice, while also providing natural language explanations for its decisions.

---------------------------------------------------------------------------------------------------
If you are interested in the detailed methodology and results of this project, please refer to our paper:
[AI-Driven Day-to-Day Route Choice](https://arxiv.org/abs/2412.03338)

---------------------------------------------------------------------------------------------------

## Example visualizations
### Example 1: Individual-level behavior of LLMTravelers

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/route_choices_of_two_agents_small.gif)  

The gif shows the route choice evolution of two LLM-gpt35-based agents (Daniel Nguyen and Jessica Ramirez) in the OW network, traveling between the same origin and destination. Initially, Daniel explored from "Route 0" to "Route 4," while Jessica moved from "Route 4" to "Route 1." Daily fluctuations in travel time influenced their perception of the average travel time for each route, leading to different learning outcomes. After 50 days, Jessica consistently chose "Route 2," while Daniel identified "Route 1" as optimal after 80 days. Their distinct exploration and memory-based decision-making led to different route choices.

---------------------------------------------------------------------------------------------------

### Example 2: Aggregate-level behavior of LLMTravelers

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/LLMTravelers_avg_tt_small.gif)  

The gif shows that all LLMTravelers converge toward a smaller travel time, indicating a move towards user equilibrium, after experiencing initial fluctuations over several days. Although the travel time becomes smaller over time, fluctuations persist, aligning with laboratory experiment findings. Additionally, the performance differences between LLM models of varying sizes are minimal. This suggests that lightweight, open-source LLMs can serve as cost-effective alternatives.

---------------------------------------------------------------------------------------------------

### Example 3: Comparison with MNL and RL-based method

![img](https://github.com/georgewanglz2019/LLMTraveler/blob/main/Diff_methods_avg_tt.gif)  

The gif shows that LLM-gpt35, MNL, and the RL-based agent converge to a similar average travel time of approximately 71.1 minutes. However, both LLM-gpt35 and the RL-based agent exhibit fluctuations around user equilibrium, achieving slightly lower travel times—behavior not observed in the MNL model. The MNL model converges more quickly in the initial days due to its method of sharing experiences, while LLM-gpt35 optimizes choices based on individual experience, which may better reflect real-world behavior. Notably, the RL-based agent requires significantly more data—nearly 4000 days—to match LLM-gpt35’s convergence within 100 days, highlighting the low sample efficiency of RL in this context, and emphasizing the comparatively higher efficiency of LLM-gpt35.

---------------------------------------------------------------------------------------------------

## Coming Soon
The full code will be released after the review process is completed.

