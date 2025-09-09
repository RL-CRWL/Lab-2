# Contributors

| Student Number | Name            |
| -------------- | --------------- |
| 2602515        | Taboka Dube     |
| 2541693        | Wendy Maboa     |
| 2596852        | Liam Brady      |
| 2333776        | Refiloe Mopeloa |

# Answers to questions

## 1.1

4. With 10,000 episodes, the value function is only an approximate estimate. 
Many states might not have been visited often enough, so the Monte Carlo estimates are noisy and can be biased. 
With 500,000 episodes, the estimates converge much closer to the true expected returns of the policy. 
The value surface is smoother, especially in less frequently visited states.

## 2.2

4. The policies are different because each method updates its Q-table differently. 
The Q-values for Q-learning are updated with 
$$
Q(S, A) = Q(S,A) + \alpha[R + \gamma max_a Q(S', a) - Q(S,A)]
$$,
whereas the Sarsa updates its values with 
$$
Q(S, A) = Q(S,A) + \alpha[R + \gamma Q(S', A') - Q(S, A)]
$$.
This means that Sarsa uses the action is will take in the next state according to its own policy, 
and Q-learning uses the best possible next action in the next state.

5. With a solely exploitative policy, both agents will choose the actions which give them the maximum reward.
Looking at the value plots, we can see that the Sarsa method has lower negative rewards further away from the cliff face. 
This means that the agent will take a longer path to get to the final destination, as it reduces the liklihood of the
agent falling.
The Q-learning method however, has lower negative rewards right up against the cliff face. This means that the Q-learning agent
will walk along the cliff as it will achieve a higher reward.
The Q-learning agent would therefore perform better as it requires fewer steps to get to the goal state.

6. Both methods will likely fail to learn the optimal policy, because they will only exploit the
state which gives them the best reward instead of exploring other states which could be better.