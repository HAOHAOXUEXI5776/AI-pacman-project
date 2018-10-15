# Pacman: Capture the Flag – Wiki
# Team: Ethereal
**Members**
1. Vitaly Yakutenko
2. Himagna Erla
3. Kartik Kishore

##### YouTube Link: https://www.youtube.com/watch?v=YDRGaf06WCo&amp=&feature=youtu.be

### https://ecotrust-canada.github.io/markdown-toc/


- [Techniques Implemented](#techniques-implemented)
  * [Approximate Q-learning](#approximate-q-learning)
    + [Approach](#approach)
    + [Bottlenecks & Issues](#bottlenecks---issues)
  * [Classical search heuristic](#classical-search-heuristic)
    + [Defensive Agent](#defensive-agent)
      - [Description:](#description-)
      - [Strategy:](#strategy-)
    + [Offensive Agent](#offensive-agent)
      - [Approach](#approach-1)
      - [Bottlenecks & Issues](#bottlenecks---issues-1)
      - [Improvements](#improvements)
  * [Model-Based MDP with Value Iteration](#model-based-mdp-with-value-iteration)
    + [Description:](#description--1)
    + [Reward Shaping](#reward-shaping)
    + [Strategy:](#strategy--1)
    + [Team of 2 Offensive Agents](#team-of-2-offensive-agents)
- [Capture the Flag Contest](#capture-the-flag-contest)
  * [Perfomance of our agents in the competetions](#perfomance-of-our-agents-in-the-competetions)
  * [Offline simulations](#offline-simulations)
    + [Intent](#intent)
    + [Set up description](#set-up-description)
    + [Results of simulations](#results-of-simulations)
    + [Problem that we can compare results against our own old teams which could be different from average team in the contest.](#problem-that-we-can-compare-results-against-our-own-old-teams-which-could-be-different-from-average-team-in-the-contest)
    + [Reasoning why we chosen Hybrid Model for the final submission](#reasoning-why-we-chosen-hybrid-model-for-the-final-submission)


# Techniques Implemented
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

### Reward shaping:
* Rewards were added to scenarios when the agent brought back food, eating a power pellet when the enemy chases us, killing an opponent agent. The same ideology of reward shaping was extended in MDP.

## Classical search heuristic

We have adopted concepts from Heuristic Search algorithms and Classical Planning to make a hybrid version of Astar search.

### Defensive Agent
#### Description:
*	The agent takes a snapshot of the defensive side of the arena for every move it makes. It also stores the immediate previous snapshot (i.e., gameState) in a variable to calculate the information it needs to define the search problem.
*	Using these snapshots, it calculates information such as the exact location of food which was just eaten by the enemy, if the enemy is observable his exact location, the nearest food location from its current position, and the entry points to our arena.
*	This agent is hybrid in a sense that the search problem and planning are defined online, that is, the goal positions and avoid positions of the agent change for based on the current game state. It performs a greedy Astar search trying to avoid the avoid positions and reach the goal positions via an optimal path. Then, the current actions are just the first action of this path returned.
*	The heuristic used is the sum of maze distances to all goal positions from the current position, and if there is an avoid position then add a value of 100 times the distance to the avoid position (i.e, we give a lot of importance to avoid positions). This heuristic is admissible, but not always optimal.

#### Strategy:
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
Based on the bottlenecks mentioned above, we configured our defense agent as an A-Star agent and found that the performance improved dramatically.


## Model-Based MDP with Value Iteration
We adopted the model based on Markov Decision Processes to derive a policy using Value-Iteration algorithmby adapting the value iteration only to a sub-grid of the map (e.g., vicinity of 7X7 grid). The most interesting application is assigning rewards to this grid, and performing reward shaping which makes the agent very efficient at what it does.

### Description:
*	First, the model was defined by taking a subgrid determined by the vicinity parameter and the transition probabilities were defined for legal actions as 1 and illegal actions as 0. For this model, a discount factor for future rewards and a number of iterations to run value iteration were assigned, and the table with zero q-values was initialised. These are the hyperparameters for the model.
*	Then, on top of this model rewards were assigned to the cells of the grid. Rewards were assigned for the objects present in the grid, including ghosts, food locations, cells with power pellets. Rewards are also assigned to some objects regardless of their presence in the grid, for example, food locations and home positions.
*	Then, value iteration with the Bellman’s equation was run at each game state for the defined model. Best action is chosen based on the derived policy from the above-mentioned process.
*	Previous steps were repeated each game step.


### Reward Shaping
*	We then, run value iteration for the model defined at each game state using the Bellman’s equation to update the values after each iteration.

### Strategy:
*	Once the agent enters the enemy arena it will try to eat the nearest food and as many as it can. The moment it spots an enemy chasing, it will avoid all corners and traps trying to run towards a power pellet.
*	If power pellet is consumed, it is very clever in a sense that it avoids eating the scared ghost and continues eating the food. If it senses that the scared timer is running out it will just eat the scared ghost.
*	If there are no power pellets in the arena and it senses that enemy is chasing it will just avoid traps and run back home to deposit the food. Provided, it has eaten enough food.
*	If any of these situations don’t occur it will eat all food until only two are left and will return back home before the end of the game.
*	Most interesting thing is that we were able to achieve this behavior by just reward shaping and assignment.

### Team of 2 Offensive Agents

We used a team consisting of 2 offensive agents based on offensive Model-Based MDP agent. While it performed well on some maps or against some teams in the contest, it had 2 main *disadvanges*:

* It ate food not efficiently enough by leaving single foods in the trap positions and going eating other food. As a result, it loosed to team with an efficient offensive agent that could eat all food without obstacles and deposit it back.
* It could not win teams with good defence. Offensive agents either was getting trapped and being eaten in some locations on the map or struggled to enter enemy's territory.



# Capture the Flag Contest
## Performance of our agents in the competitions
As the competition was run once daily, we found it necessary to test our model for improvements against older versions. The same has been included in TestingScripts section of our git-lab repository.


## Offline simulations & Performance check

### Intent
* To test out how well the new agent performed on different terrains and scenarios where it lost in earlier simulations and daily competitions.
* Ensure that every iteration of code is first tested against a set of teams and maps so as to achieve consistency in performance and then deployed.
* A script was written to evaluate the performance of new agent against older and different implementations.

### Set up description
* Combination 100 fixed and random layouts, we kept increasing the sample size with the layouts that were being run on the competitions everyday so as to increase the versatility of the evaluation.

### Results
* Tune the Hyper-parameters in our code.
* Through this we were able to rectify fail states if any.
* Improve strategy, based on agent’s performance.

> Numbers, data and plots comes here



From the above, we can say that MDP fared better against our earlier models but was falling short against baselineTeam, whereas our Hybridv4 model, that encompasses the offense of an MDP agent and the defense of an A-Star agent was best suited for a generic approach against the competition, moreover this idea was supported by the everyday evaluation for both the models.


# Final Model
`MDP-based offensive Agent || A-Star-base defensive agent`
### Reason:
With respect to the above results, we can say that MDP fared better against our earlier models but was falling short against baselineTeam, whereas our HybridV4 and HybridV5 model, that encompasses the offense of an MDP agent and the defense of an A-Star agent was best suited for a generic approach against an average competition team, moreover this idea was supported by the everyday evaluation during the last week of competitions.
The MDP version 1-2 were performing consistently and winning against Hybrid V1 on a scenario of maps but the MDP failed against Baseline agents on a number of scenarios. When we submitted the MDP agent it performed decent on the competition, achieving a rank of 20 among 90 teams, but as the competition progressed on, the MDP was not able to cope up with ever changing teams. So we then decided to change the primary offensive agent to MDP and the defensive agent as A-Star, and thus the hybrid model was created. Improvements have been made over time to tweak the performance and behaviour of both the agents and have been termed as Hybrid<version_number>

# Future Improvements
1. <B>MDP</B> can be further tuned by a better reward shaping function, right now we are using a linear scale to shape rewards, we tried doing a few exponential approaches but could not implement it due to shortage of time. An exponential reward shaping will be better for quick decisions on the fly with lesser ambiguity.
2. <B>A-STAR</B> is a very versatile model but the heuristic of goal and avoid positions can be improved, a better strategy would be to design heuristics based on the study of the arena.

### Problem that we can compare results against our own old teams which could be different from average team in the contest.
### Reasoning why we chosen Hybrid Model for the final submission
