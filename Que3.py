from openai import OpenAI
import torch
import openai
import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils

from Que2 import mdp_solver_q_learning

from minihouse.robotminihousemodel import MiniHouseV1
from minihouse.minihousev1 import test_cases

# ------- your code starts here ----- #
client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = "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"
)
# ------- your code ends here ------- #


class LLM_Model:
    def __init__(self, device, instruction, goal_state, initial_conditions, model = "meta-llama/llama-2-13b-chat"):
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
        
        if not action_list or not action_index:
            print("⚠️ Warning: No valid LLM response. Using fallback action.")
            return [self.GROUNDED_ACTION_LIST[0], 0]  # fallback to first action
        
        best_action = [max(action_list, key=action_list.count), max(action_index, key=action_list.count)]
        # [action, index]
        return best_action
    
    def query_llm(self, task, observe = None, prompt = None, env_condition = None):
        prompt_content = self.prompt_begin
        task = 'Scene: ' + "\n".join(f"{a}:{b}" for a, b in zip(self.condition_list, self.initial_conditions) if a != 'ACTION_DICT') + '\nCurrent State: ' + observe + 'Task: ' + task

        
        generated_samples = []
        for _ in range(self.sampling_params['n']): # call llm for n times and append each response to 'generated_samples'
            try: 
                
                # ------- your code starts here ----- #
                
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.prompt_begin},
                        {"role": "user", "content": task}
                    ],
                    **self.sampling_params
                )

                # Extract the message content
                generated_samples.append(response.choices[0].message.content.strip())
                
                # ------- your code ends here ------- #
                
            except Exception as e:
                print(f"Error: {e}")

        samples = generated_samples
        # print(samples)
        return samples

def describe_state(state):
    describe = ''

    # ------- your code starts here ----- #

    describe = f"The robot is currently in the {state.robot_agent.position}.\n"

    for obj_name, obj in state.object_dict.items():
        describe += f"The {obj_name} is located at {obj.position}.\n"

    for container_name, container in state.container_dict.items():
        describe += f"The {container_name} is {'open' if container.is_open else 'closed'}.\n"

    for surface_name, surface in state.surface_dict.items():
        describe += f"The {surface_name} is in the {surface.position}.\n"
            
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
                          model = "meta-llama/llama-2-13b-chat")

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


if __name__ == "__main__":
    test_q_learning_with_llm(index=0, num_episodes=1000, verbose=True)