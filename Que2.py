import numpy as np

from minihouse.robotminihousemodel import MiniHouseV1
from minihouse.minihousev1 import test_cases

class mdp_solver_q_learning:
    
    def __init__(self):
        print("MDP initialized for Q-learning!")


    def epsilon_greedy(self, Q, state, epsilon):
        if np.random.rand() < epsilon:
            
            # ------- your code starts here ----- #
            return np.random.choice(Q.shape[1])          
            # ------- your code ends here ------- #
            
        else:
            
            # ------- your code starts here ----- #
            return np.argmax(Q[state])
            # ------- your code ends here ------- #
   

    def Q_learning(
        self, alpha:float, gamma:float, theta:float, epsilon:float, env_nS:int, env_nA:int, env_transition, env, num_episodes=1000, initial_action = None, initial_reward = 10):
        """
        Q-learning algorithm.
        Args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            num_episodes: number of episodes
        Returns:
            Q: learned Q-value function
            rewards: rewards obtained in each episode
        """
        Q = np.zeros((env_nS, env_nA))
        rewards = []
        
        if initial_action is not None:
            for i, action in enumerate(initial_action):
                Q[i][action] += initial_reward
        
        for episode in range(num_episodes):
            env.reset()
            state = env.state_to_index(env.state)
            done = False
            
            while not done:
                
                if episode == 0:
                    Q_prev = Q.copy()
                if 'total_reward' not in locals():
                    total_reward = 0

                action_idx = self.epsilon_greedy(Q, state, epsilon)
                action = env.index_to_action(action_idx)
                _, reward, done, _, _ = env.step(action)
                next_state_idx = env.state_to_index(env.get_state)

                best_next_action = np.argmax(Q[next_state_idx])
                Q[state][action_idx] += alpha * (reward + gamma * Q[next_state_idx][best_next_action] - Q[state][action_idx])

                state = next_state_idx

                # Accumulate rewards in the current episode
                if len(rewards) <= episode:
                    rewards.append(reward)
                else:
                    rewards[episode] += reward

                # Check for convergence after the episode ends
                if done:
                    if episode > 0 and np.max(np.abs(Q - Q_prev)) < theta:
                        rewards = rewards[:episode+1]  # truncate to valid episodes
                        done = True
                        raise StopIteration  # force exit from outer loop
                    Q_prev = Q.copy()
                    total_reward = 0

                    
        return np.argmax(Q, axis=1), rewards
    

def test_q_learning(Q_bool=False, index=0, num_episodes=1000, verbose=False, generate_solution=False):

    instruction, goal_state, initial_conditions = test_cases[index]

    env = MiniHouseV1(
        instruction=instruction,
        goal_state=goal_state,
        initial_conditions=initial_conditions,
        verbose=verbose
    )
    env.reset()
    
    
    if verbose:
        print("state: ", env.state_to_index(env.state))
        print("num state: ", env.nS)
        print("num actions: ", env.nA)
        print()
        
    msq = mdp_solver_q_learning()


    policy, V = msq.Q_learning(
        alpha=0.1,
        gamma=0.9,
        theta=0.0001,
        epsilon=0.1,
        env_nS=env.nS,
        env_nA=env.nA,
        env_transition=env.transition,
        env=env,
        num_episodes=num_episodes,
    )

    print("Learned policy:")
    print(policy)

    
    if verbose:
        print("Policy Iteration")
        print()
        print("V: ", repr(V))
        print()
        print("policy: ", repr(policy))
        print()
        
    env.reset()
    
    for i in range(100):
        print()
        print(f"---------- Step: {i} ----------")
        action = int(policy[env.state_to_index(env.state)])
        # action = policy[env.state_to_index(env.state)].astype(int)
        obs, reward, done, _, _ = env.step(action)
        if verbose:
            print("obs: ", obs)
            print("reward: ", reward)
            print("done: ", done)
        if done:
            break

    if generate_solution:
        np.savetxt(f'data/V_{index}.py', V, delimiter=',')

        solution_values = np.loadtxt(f'data/V_{index}.py')

        assert len(V) == len(solution_values), \
            'Length of Values is incorrect'

        assert np.allclose(V, solution_values), \
            'Values incorrect'

    return None

if __name__ == "__main__":
    test_q_learning(index=0, num_episodes=1000, verbose=True)
