import numpy as np

from typing import Callable
from minihouse.robotminihousemodel import MiniHouseV1
from minihouse.minihousev1 import test_cases

class mdp_solver:
    
    def __init__(self):
        self.iteration = 0
        print("MDP initialized!")


    def get_action_value(
        self, s:int, a:int, V:np.ndarray, gamma:float, env_transition:Callable):
        r"""
        Code for getting action value. Compute the value of taking action a in state s
        I.e., compute Q(s, a) = \sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
        args:
            s: state
            a: action
            V: value function
            gamma: discount factor
            env_transition: transition function
        returns:
            value: action value
        """
        value = 0

        for prob, next_state, reward, done in env_transition(s, a):

            # ------- your code starts here ----- #

            value += prob * (reward + gamma * V[next_state])

            # ------- your code ends here ------- #

        return value



    def get_max_action_value(
        self, s:int, env_nA:int, env_transition:Callable, V:np.ndarray, gamma:float):
        r"""
        Code for getting max action value. Takes in the current state and returns 
        the max action value and action that leads to it. I.e., compute
        a* = argmax_a \sum_{s'} p(s'| s, a) * [r + gamma * V(s')]
        args:
            s: state
            env_nA: number of actions
            env_transition: transition function
            V: value function
            gamma: discount factor
        returns:
            max_value: max action value
            max_action: action that leads to max action value
        """
        max_value = -np.inf
        max_action = -1

        for a in range(env_nA):

            # ------- your code starts here ----- #

            value = self.get_action_value(s, a, V, gamma, env_transition)
            if value > max_value:
                max_value = value
                max_action = a

            # ------- your code ends here ------- #

        return max_value, max_action



    def get_policy(
        self, env_nS:int, env_nA:int, env_transition:Callable, gamma:float, V:np.ndarray):
        """
        Code for getting policy. Takes in an Value function and returns the optimal policy
        args:
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            gamma: discount factor
            V: value function
        returns:
            policy: policy
        """
        policy = np.zeros(env_nS)

        for s in range(env_nS):
            max_value = -np.inf
            max_action = -1
            for a in range(env_nA):

                # ------- your code starts here ----- #
            
                value = self.get_action_value(s, a, V, gamma, env_transition)
                if value > max_value:
                    max_value = value
                    max_action = a

                # ------- your code ends here ------- #

            policy[s] = max_action

        return policy
        


    def policy_evaluation(
        self, env_nS:int, env_transition:Callable, V:np.ndarray, gamma:float, theta:float, policy:np.ndarray):
        """
        Code for policy evaluation. Takes in an MDP and returns the converged value function
        args:
            env_nS: number of states
            env_transition: transition function
            V: value function
            gamma: discount factor
            theta: convergence threshold
            policy: policy
        returns:
            V: value function
        """ 

        while True:
            delta = 0
            for s in range(env_nS):

                # ------- your code starts here ----- #

                v = 0
                for prob, next_state, reward, done in env_transition(s, policy[s]):
                    v += prob * (reward + gamma * V[next_state])
                delta = max(delta, abs(V[s] - v))
                V[s] = v

                # ------- your code ends here ------- #

            if delta < theta:
                break

        return V
        


    def policy_improvement(
        self, env_nS:int, env_nA:int, env_transition:Callable, policy:np.ndarray, V:np.ndarray, gamma:float):
        """
        Code for policy improvement. Takes in an MDP and returns the converged policy
        args:
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
            policy: policy
            V: value function
            gamma: discount factor
        returns:
            policy_stable: whether policy is stable
            policy: policy
        """
        policy_stable = True

        for s in range(env_nS):

            # ------- your code starts here ----- #

            old_action = policy[s]
            max_value = -np.inf
            best_action = -1
            for a in range(env_nA):
                value = self.get_action_value(s, a, V, gamma, env_transition)
                if value > max_value:
                    max_value = value
                    best_action = a
            policy[s] = best_action
            if old_action != best_action:
                policy_stable = False

            # ------- your code ends here ------- #

        return policy_stable, policy



    def value_iteration(
        self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
        """
        The code for value iteration. Takes in an MDP and returns the optimal policy
        and value function.
        args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
        returns:
            policy: optimal policy
            V: optimal value function 
        """
        V = np.zeros(env_nS)
        converged = False

        while not converged:
            delta = 0

            # ------- your code starts here ----- #

            for s in range(env_nS):
                max_value = -np.inf
                for a in range(env_nA):
                    value = self.get_action_value(s, a, V, gamma, env_transition)
                    if value > max_value:
                        max_value = value
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value
            if delta < theta:
                converged = True

            # ------- your code ends here ------- #

        policy = self.get_policy(env_nS, env_nA, env_transition, gamma, V)
        
        return policy, V
        


    def policy_iteration(
        self, gamma:float, theta:float, env_nS:int, env_nA:int, env_transition:Callable):
        """
        Code for policy iteration. Takes in an MDP and returns the optimal policy
        args:
            gamma: discount factor
            theta: convergence threshold
            env_nS: number of states
            env_nA: number of actions
            env_transition: transition function
        returns:
            policy: optimal policy
            V: optimal value function
        """
        V = np.zeros(env_nS)
        policy = np.zeros(env_nS)
        converged = False

        while not converged:

            # ------- your code starts here ----- #

            while not converged:
                V = self.policy_evaluation(env_nS, env_transition, V, gamma, theta, policy)
                policy_stable, policy = self.policy_improvement(env_nS, env_nA, env_transition, policy, V, gamma)
                if policy_stable:
                    converged = True

            # ------- your code ends here ------- #

        return policy, V


def test(VI_bool = True, index = 0, verbose=False):

    instruction, goal_state, initial_conditions = test_cases[index]

    env = MiniHouseV1(
        instruction=instruction,
        goal_state=goal_state,
        initial_conditions=initial_conditions,
        verbose=verbose,
    )
    env.reset()
    if verbose:
        print("state: ", env.state_to_index(env.state))
        print("num state: ", env.nS)
        print("num actions: ", env.nA)
        print()

    ms = mdp_solver()
    
    if VI_bool:
        policy, V = ms.value_iteration(0.9, 0.0001, env_nS=env.nS, env_nA=env.nA, env_transition=env.transition)
        if verbose:
            print("Value Iteration")
            print()
    else:
        policy, V = ms.policy_iteration(0.9, 0.0001, env_nS=env.nS, env_nA=env.nA, env_transition=env.transition)
        if verbose:
            print("Policy Iteration")
            print()
    
    if verbose:
        print("V: ", repr(V))
        print()
        print("policy: ", repr(policy))
        print()
    
    env.reset()
    
    for i in range(100):
        print()
        print(f"---------- Step: {i} ----------")
        
        action = int(policy[env.state_to_index(env.state)])
        obs, reward, done, _, _ = env.step(action)
        
        if verbose:
            print("obs: ", obs)
            print("reward: ", reward)
            print("done: ", done)
        
        if done:
            break


if __name__ == "__main__":
    test(index=0, VI_bool=True, verbose=True)
