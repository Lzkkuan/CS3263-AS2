"""
Assignment 2 programming script.

* Group Member 1:
    - Name:
    - Student ID:

* Group Member 2:
    - Name:
    - Student ID:
"""


import numpy as np

from typing import Callable



# Assignmen 2: Policy Iteration & Value Iteration

def get_action_value(
    self, s:int, a:int, V:np.ndarray, gamma:float, env_transition:Callable):
    """
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

        

        # ------- your code ends here ------- #

    return value



def get_max_action_value(
    self, s:int, env_nA:int, env_transition:Callable, V:np.ndarray, gamma:float):
    """
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

        

        # ------- your code ends here ------- #

    return policy, V



# Assignmen 2: Q-Learning

def epsilon_greedy(self, Q, state, epsilon):
    if np.random.rand() < epsilon:

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #

    else:

        # ------- your code starts here ----- #

        

        # ------- your code ends here ------- #



def Q_learning(
    self, alpha:float, gamma:float, theta:float, epsilon:float, env_nS:int, env_nA:int, env_transition, env, num_episodes=1000):
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

    for episode in range(num_episodes):
        env.reset()
        state = env.state_to_index(env.state)
        done = False

        while not done:

            # ------- your code starts here ----- #

            

            # ------- your code ends here ------- #

    return np.argmax(Q, axis=1), rewards


# Assignmen 3: Q-Learning with LLM

# ------- your code starts here ----- #
client = OpenAI(
    base_url = "",
    api_key = "Your api key"
)
# ------- your code ends here ------- #


class LLM_Model:
    def __init__(self, device, instruction, goal_state, initial_conditions, model='gpt-3.5-turbo'):
        self.device = device
        self.model = model
        self.sampling_params = \
            {
                "max_tokens": 32,
                "temperature": 0.5,
                "top_p": 0.9,
                "n": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": ['\n']
            }
        
        self.prompt_begin = """Generate the most logical next move in the scene. 
You must strictly follow the format in the following examples and output exactly **one** action. 
The generated action must be strictly from **GROUNDED_ACTION_LIST**."""
        self.condition_list = ['ROOM_LIST', 'OBJECT_LIST', 'OBJECT_POSITION_LIST','CONTAINER_LIST', 'SURFACE_LIST', 'CONTAINER_POSITION_LIST', 'CONNECTED_ROOM','ACTION_DICT', 'GROUNDED_ACTION_LIST']
        self.initial_conditions = initial_conditions
        self.ROOM_LIST, self.OBJECT_LIST, self.OBJECT_POSITION_LIST, \
        self.CONTAINER_LIST, self.SURFACE_LIST, self.CONTAINER_POSITION_LIST, self.CONNECTED_ROOM, \
        self.ACTION_DICT, self.GROUNDED_ACTION_LIST = initial_conditions
        
        self.translation_lm = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        self.container_list_embedding = self.translation_lm.encode(self.CONTAINER_LIST, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory
        self.object_list_embedding = self.translation_lm.encode(self.OBJECT_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
        if self.SURFACE_LIST:
            self.position_list =  self.CONTAINER_LIST + self.SURFACE_LIST
            self.surface_list_embedding = self.translation_lm.encode(self.SURFACE_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
            self.position_list_embedding = torch.concat((self.container_list_embedding, self.surface_list_embedding), dim=0)
        self.room_embedding = self.translation_lm.encode(self.ROOM_LIST, batch_size=8,
                convert_to_tensor=True, device=self.device)
        self.action_list_embedding = self.translation_lm.encode(self.GROUNDED_ACTION_LIST, batch_size=8, 
                convert_to_tensor=True, device=self.device)  # lower batch_size if limited by GPU memory
    
    def find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        most_similar_idx = np.argmax(cos_scores)
        return most_similar_idx
    
    def plan(self, task, observe, env_condition = None, prompt = None):
        samples = self.query_llm(task, observe = observe, prompt = prompt, env_condition = env_condition)
        action_list, action_index = [], []
        for sample in samples:
            most_similar_idx = self.find_most_similar(sample, self.action_list_embedding)
            translated_action = self.GROUNDED_ACTION_LIST[most_similar_idx]
            action_list.append(translated_action)
            action_index.append(most_similar_idx)
        best_action = [max(action_list, key=action_list.count), max(action_index, key=action_list.count)]
        # [action, index]
        return best_action
    
    def query_llm(self, task, observe = None):
        prompt_content = self.prompt_begin
        task = 'Scene: ' + "\n".join(f"{a}:{b}" for a, b in zip(self.condition_list, self.initial_conditions) if a != 'ACTION_DICT') + '\nCurrent State: ' + observe + 'Task: ' + task

        
        generated_samples = []
        for _ in range(self.sampling_params['n']): # call llm for n times and append each response to 'generated_samples'
            try: 
                
                # ------- your code starts here ----- #
                
                
                
                
                # ------- your code ends here ------- #
                
            except Exception as e:
                print(f"Error: {e}")

        samples = generated_samples
        # print(samples)
        return samples

def describe_state(state):
    describe = ''

    # ------- your code starts here ----- #
    
    
    
    
    
    
    
    # ------- your code ends here ------- #
    
    
    print(describe)
    return describe

def test_q_learning_with_llm(Q_bool=False, index=0, num_episodes=1000, verbose=False, generate_solution=False):

    instruction, goal_state, initial_conditions = test_cases[index]
    
    # ------- your code starts here ----- #

    llm_model = LLM_Model(device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                          instruction=instruction,
                          goal_state=goal_state,
                          initial_conditions=initial_conditions, 
                          model = '')

    # ------- your code ends here ------- #

    
    env = MiniHouseV1(
        instruction=instruction,
        goal_state=goal_state,
        initial_conditions=initial_conditions,
        verbose=verbose
    )
    obs, reward, done, history, valid_action = env.reset()

    best_action_llm = []
    print('Your task is:' + instruction)
    for state_index in range(env.nS):
        print("state: ", state_index)
        describe = describe_state(env.index_to_state(state_index))
        best_action = llm_model.plan(instruction, observe=describe)
        action_name = best_action[0]
        action_index = best_action[1]
        best_action_llm.append(action_index)
        print('The best action is ' + action_name + '\n')
    
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
        initial_action = best_action_llm,
    )
    
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