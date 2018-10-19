# Pacman: Capture the Flag – Wiki
# Team: Ethereal
**Members**
1. Vitaly Yakutenko
2. Himagna Erla
3. Kartik Kishore

**YouTube Link: https://www.youtube.com/watch?v=eDfJ4MxW69c&feature=youtu.be**

# Table of Contents

- [Introduction](#introduction)
- [Techniques Attempted](#techniques-attempted)
  * [Approximate Q-learning](#approximate-q-learning)
  * [Heuristic Search Algorithm](#heuristic-search-algorithm)
  * [Model-Based MDP with Value Iteration](#model-based-mdp-with-value-iteration)
- [Capture the Flag Contest](#capture-the-flag-contest)
  * [Performance of our agents in the competitions](#performance-of-our-agents-in-the-competitions)
  * [Offline simulations & Performance check](#offline-simulations---performance-check)
- [Final Model](#final-model)
- [Future Improvements](#future-improvements)

# Introduction
In this project, we attempted 3 different techniques: Approximate Q-learning, Heuristic Search, Model-Based MDP. For the final submission we selected Offensive agent based on MDP model and for defensive, we opted for heuristic search agent. 

# Techniques Attempted
## Approximate Q-learning
### Approach
*	Based on a set of features and weights, an action was chosen with the maximum Q value, similar to Baseline-Team.
*	Tried training the agent against older implementations with initial values assigned to enable bootstrapping but the model could not learn the weights properly.
*	Hyperparameters α and γ were modified to see how the model performed in the various scenario.
*	Primary objective with this type of model was to attack and eat greedily.

### Bottlenecks & Issues
*	Had issues with training, the end result was always the training weights getting approximated to the initial weights that we had bootstrapped them with. We tried to make the model learn slower but still were not able to achieve a fully functional learning model.
*	The performance of the agent was good in layouts with multiple entry points into the enemy arena, but it failed on restricted layouts.
*	The approach failed against teams which had a strong defense which forced us to move to another approach.
* What we believe to be issues in reward shaping, resulted in getting garbage values for the weights assigned to scenarios.

### Reward shaping
* Rewards were added to scenarios when the agent brought back food, eating a power pellet when the enemy chases us, killing an opponent agent. The same ideology of reward shaping was extended in MDP.

## Heuristic Search Algorithm

We have adopted concepts from Heuristic Search algorithms and Classical Planning to make a hybrid version of Astar search.

### Defensive Agent
#### Description
*	The agent takes a snapshot of the defensive side of the arena for every move it makes. It also stores the immediate previous snapshot (i.e., gameState) in a variable to calculate the information it needs to define the search problem.
*	Using these snapshots, it calculates information such as the exact location of food which was just eaten by the enemy, if the enemy is observable his exact location, the nearest food location from its current position, and the entry points to our arena.
*	This agent is hybrid in a sense that the search problem and planning are defined online, that is, the goal positions and avoid positions of the agent change for based on the current game state. It performs a greedy Astar search trying to avoid the avoid positions and reach the goal positions via an optimal path. Then, the current actions are just the first action of this path returned.
*	The heuristic used is the sum of maze distances to all goal positions from the current position, and if there is an avoid position then add a value of 100 times the distance to the avoid position (i.e, we give a lot of importance to avoid positions). This heuristic is admissible, but not always optimal.

#### Strategy
*	If the current game state is such that there is no enemy Pacman in our arena the only goal positions of the Astar agent is to patrol around the entry points of the arena trying to not let the enemy enter.
*	The moment it senses a food missing the immediate goal is to go to that last missing food location. If on the way it finds the enemy Pacman and provided it's not scared it will chase after the enemy.
*	If our agent is in a scared state, the goal positions now are to go to the food missing location and trying to avoid the getting eaten by the enemy Pacman provided it is not too far away from the start allowing it roam around the enemy until it becomes a ghost again to stop it. This strategy was developed after running many random matches and simulation against our own old versions as mentioned in below sections.
*	Finally, if it happens such that Astar has returned a null path or an illegal action we have adopted the agent such that it will take action by using the Baseline defensive agent Q-values. This might be a rare case, but this allows our agent to take more rational actions rather than taking random actions when Astar returns null.

### Offensive Agent
#### Approach
*	The model was based on basic search techniques discussed in classical planning during the lectures.
*	A-Star algorithm was used for searching by the offensive agent. The heuristic that was passed to the search agent was a combination of goal positions and avoid positions. A basic approach in nature, but this helped us to refine our strategy a lot for the daily competitions.
*	We observed how the agent behaved in scenarios and modified the heuristic based on the game. Few tweaks we did:
 - The goal positions for the offensive agent were the enemy food pellets and power pellets.
 - If the enemy ghost was observable, the goal position was changed to enemy’s power pellets, so the agent could eat more when the enemy ghost was scared, keeping track of the scared timer and eat the ghost when the timer is about to end.
 - Avoid going to corners in the grid.
*	This model could be tuned with ease and improved our efficiency in the competition, all the while we were using the defense agent of the baseline Team and playing around with parameters.

#### Bottlenecks & Issues
*	The A-Star agent could not perform against teams which had a strong defense.
*	The agent still went into areas which had only one entry point and thus was cornered by the enemy’s defense agent.
*	Defence agent was not adept in tackling the enemy in various scenarios.

#### Improvements
Based on the bottlenecks mentioned above, we configured our defence agent as an A-Star agent and found that the performance improved dramatically.


## Model-Based MDP with Value Iteration
We adopted the model based on Markov Decision Processes that derives a policy using Value Iteration algorithm. In order to apply this approach, a sub-grid around the agent was generated (e.g., vicinity of 7X7 grid) at each step of the game. Then, rewards were assigned for certain cells of the resulting grid and value iteration was run. The most interesting and challenging part was assigning the relevant rewards to help value iteration algorithm to derive a good policy.

### Description
*	First, the model was defined by taking a subgrid determined by the vicinity parameter and the transition probabilities were defined for legal actions as 1 and illegal actions as 0. For this model, a discount factor for future rewards and a number of iterations to run value iteration were assigned, and the table with zero Q-values was initialized. 
*	Then, on top of this model rewards were assigned to the cells of the grid. Rewards were assigned for the objects present in the grid, including ghosts, food locations, cells with power pellets. Rewards are also assigned to some objects regardless of their presence in the grid, for example, food locations and home positions.
*	Then, value iteration with the Bellman’s equation was run at each game state for the defined model. Best action was chosen based on the derived policy from the above-mentioned process.
*	Previous steps were repeated each game step.

### Strategy Used
For this model, both agents were offensive with a goal of eating all food on enemy's territory and depositing it back on home territory.
*	Once the agent enters the enemy arena it will try to eat the nearest food and as many as it can. The moment it spots an enemy chasing, it will avoid all corners and traps trying to run towards a power pellet.
*	If power pellet is consumed, it is very clever in a sense that it avoids eating the scared ghost and continues eating the food. If it senses that the scared timer is running out it will just eat the scared ghost.
*	If there are no power pellets in the arena and it senses that enemy is chasing it will just avoid traps and run back home to deposit the food. Provided, it has eaten enough food.
*	If any of these situations don’t occur it will eat all food until only two are left and will return back home before the end of the game.
*	Most interesting thing is that we were able to achieve this behavior in most of the cases.

### Assigning Rewards
Rewards were chosen in order to accomplish a goal of eating all food and safely deposit it back depending on the current context of the agent. Rewards were assigned to different cells of the current MDP grid for objects inside of the MDP-grid as well as objects outside it. Different strategies for reward assignment were implemented and will be described below.  

**Reward types:**
* Food - small positive rewards were used for cells with food dots and shaped rewards for cells leading to the food. 
* Other agents' positions rewards - high negative rewards were assigned to the cells with not scared ghosts and small negative reward for teammate agent position.
* Power pellets positions rewards and shaped rewards for cells leading to power pellet positions. Depending on context rewards could be positive or negative.
* Trap positions - negative rewards were assigned to cells with single legal action 
* Going home rewards were assigned to cells leading to home territory.

### Reward assignment strategies
Several strategies were implemented to allocate rewards to MDP-grid.

#### Standard Rewards Shaping
* All cells in the MDP-grid were given some shaped reward for all relevant objects in the map. For example, for all enemy's food dots a constant positive reward was divided by distance and added to the reward for all cells inside of the grid.
* On the image below could be seen rewards that were allocated to the grid of the orange Pacman. It contains a lot of positive (green) shaped rewards for food inside of 'visible' grid as well as for food outside of it'. Also, negative rewards for the position of teammate Pacman could be seen.
![image](/uploads/e3027fc620b67d0f50530418e64d1021/image.png)

* Another example of given rewards could be seen on the image below. Ghost was chasing red team pacmans, so negative rewards were given for ghost position and shaped positive rewards for the power pellet. 
![image](/uploads/67d18700f53d3f64dffb43287bdac675/image.png)
* We made attempts to change the shape function, but could not overcome the issues below.

**Issues with that strategy**:
* Agents often left single food next to them and moved towards locations with food clusters. So the consuming of food was not efficient enough and enemy's Pacman could eat red teams' food and deposit it faster.
* Sometimes MDP-agent was trapped in locations with a single point of entrance and got killed when ghost occasionally came across it - referred to as trap locations in the video.
* Sometimes the agent struggled to enter enemy's territory if some defensive agent was defeating the entrance. 

#### Assigning rewards for boundary cells
* Then we tried to assign rewards for objects inside of the grid, but outside of the grid give some cumulative reward to the boundary cell of MDP. Rewards yielded could be seen on the following picture for a situation when agents were approaching the enemy's field.
![image](/uploads/68e822eb22d143e5f2aca68ebdd07932/image.png)
* On this picture could be seen recast rewards for food. Also, scared ghost, the whose scared timer is about to expire yields negative rewards for its position.
![image](/uploads/3c1823f5016e1cee51e5fba7d00172a9/image.png)
As a result, food consuming has become more efficient, but we had no time to finish this approach because of the assignment deadline.
* On this picture could be seen red Pacman running away from the ghost. Purple cells represent an area that was expanded with depth=5 and all of them gave a small number of cells. If we had had more time, we would have implemented a search algorithm in order to find an exit and set correct rewards. Unfortunately, in this particular case red Pacman headed to 2 food dots and was eaten.
![image](/uploads/06d57e28dcba9627a6d567b10fab25fd/image.png)


### Summary of MDP - Value Iteration approach
**Advantages:**
* Derived MDP with rewards could be reused for Q-learning, Sarsa or n-step TD Learning.

**Disadvantages:**
* Painstaking process of assigning rewards.
* Elaborate tunning of hyperparameters required in order to achieve the desired performance.
* Approach with 2 offensive agents proved to be vulnerable against teams with good both defence and attack. They prevented MDP agents from eating enough food, while opponent offensive agent without any hurdles ate all food and deposited it back home as there was no defence of our end.


# Capture the Flag Contest
## Performance of our agents in the competitions
As the competition was run once daily, we found it necessary to test our model for improvements against older versions. The same has been included in TestingScripts section of our git-lab repository.


## Offline simulations & Performance check

### Intent
* To test out how well the new agent performed on different terrains and scenarios where it lost in earlier simulations and daily competitions.
* Ensure that every iteration of code is first tested against a set of teams and maps so as to achieve consistency in performance and then deployed.
* A script was written to evaluate the performance of new agent against older and different implementations.

### Set up description
* Combination 100 fixed and random layouts, we kept increasing the sample size with the layouts that were being run on the competitions every day so as to increase the versatility of the

### Results
* Tune the Hyper-parameters in our code.
* Through this we were able to rectify fail states if any.
* Improve strategy, based on agent’s performance.

![image](/uploads/5737c63d44c98c3cabf743442416b132/image.png)
* Hybrid_v - Combination of MDP and A-Star - Version number indicate changes to A-Star heuristic and MDP inclusions over time.
* MDP_v - Minor tweaks over time on Value Iteration Hyperparameters

![image](/uploads/5b1d3e0adb307dd4077e9c91926019e5/image.png)

From the above, we can say that MDP fared better against our earlier models but was falling short against baselineTeam, whereas our Hybridv4 model, that encompasses the offense of an MDP agent and the defense of an A-Star agent was best suited for a generic approach against the competition, moreover this idea was supported by the everyday evaluation for both the models.


# Final Model
`MDP-based offensive Agent || A-Star-base defensive agent`
### Reason
* With respect to the above results, we can say that MDP fared better against our earlier models but was falling short against baselineTeam, whereas our HybridV4 and HybridV5 model, that encompasses the offense of an MDP agent and the defense of an A-Star agent was best suited for a generic approach against an average competition team, moreover this idea was supported by the everyday evaluation during the last week of competitions.
* The MDP version 1-2 were performing consistently and winning against Hybrid V1 on a scenario of maps but the MDP failed against Baseline agents on a number of scenarios. When we submitted the MDP agent it performed decent on the competition, achieving a rank of 20 among 90 teams, but as the competition progressed on, the MDP was not able to cope up with ever changing teams. So we then decided to change the primary offensive agent to MDP and the defensive agent as A-Star, and thus the hybrid model was created. Improvements have been made over time to tweak the performance and behaviour of both the agents and have been termed as Hybrid_version_number.
* `NOTE:` The final submission in myTeam.py is that of Hybrid4

# Future Improvements
1. <B>MDP</B> can be further tuned by a better reward shaping function, right now we are using a linear scale to shape rewards, we tried doing a few exponential approaches but could not implement it due to a shortage of time. An exponential reward shaping will be better for quick decisions on the fly with lesser ambiguity.
2. <B>A-STAR</B> is a very versatile model but the heuristic of goal and avoid positions can be improved, a better strategy would be to design heuristics based on the study of the arena.