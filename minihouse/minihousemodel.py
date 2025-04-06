import numpy as np
import copy
from typing import Dict, Tuple
import random

IN_ROBOT_HAND = "picked by you"

# TODO: table could not be opened

class item_object:
    def __init__(self, name: str, is_moveable: bool, position: str) -> None:
        self.name = name
        self.position = position
        self.is_moveable = is_moveable

class key_object:
    def __init__(self, name: str, room, container) -> None:
        self.name = name
        self.room = room
        self.container = container
        self.in_robot_hand = False

class room_class:
    def __init__(self, key, is_open, name, connected_rooms, is_locked) -> None:
        self.name = name
        self.connected_rooms = connected_rooms
        self.is_locked = is_locked
        self.key = key
        self.is_open = is_open

class surface_object:
    def __init__(self, name: str, position: str, ) -> None:
        self.name = name
        self.position = position

class container_object:
    def __init__(self, name: str, key: key_object, is_open: bool, 
                position: str, is_locked: bool) -> None:
        self.name = name
        self.position = position
        self.is_open = is_open
        self.is_locked = is_locked
        self.key = key

class state_class:
    def __init__(self, object_list, room_list, container_list, surface_list, robot_agent, human_agent) -> None:
        self.object_dict = object_list
        self.container_dict = container_list
        self.surface_dict = surface_list
        self.room_dict = room_list
        self.robot_agent = robot_agent
        self.human_agent = human_agent

    def get_room(self, room_name):
        return self.room_dict[room_name]
    
    def get_surface(self, surface_name):
        return self.surface_dict[surface_name]

    def get_container(self, container_name):
        return self.container_dict[container_name]
    
    def get_item(self, item_name):
        return self.object_dict[item_name]
    
    def get_key(self, key_name):
        return self.object_dict[key_name]
    
    def get_robot(self):
        return self.robot_agent
    
    def get_human(self):
        return self.human_agent

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, state_class):
            for object in self.object_dict.values():
                if object.name not in __value.object_dict:
                    print(object.name)
                    return False
                if object.position != __value.object_dict[object.name].position:
                    print(object.name)
                    print(object.position)
                    print(__value.object_dict[object.name].position)
                    return False
        return True 
    
    def copy(self):
        return copy.deepcopy(self)

class observation_class:
    '''Observation class for minihouse environment
        If the robot is at a room, it will be able to observe the containers in the room, and items not in containers.
            If the containers are open, the robot is also able to observe the items in the containers.
        If the observation contradicts the state, the state will be updated to match the observation. That consists of
            two possible cases: 
            1. The items or containers are missing from the observation, but are present in the state. In this case, the
                items and containers will be removed from the room, and the items and containers' position will be set according
                to the prior knowledge of the environment, which is from large language models. 
            2. The items or containers are present in the observation, but are missing from the state. In this case, the items
                and containers will be added to the room, and the items and containers' position will be updated according to 
                the observation. 
    '''
    def __init__(self, room, container_dict, item_dict) -> None:
        '''only observe the containers and items in the room'''
        self.room_obj = room
        if self.room_obj is not None:
            self.room = room.name
        else:
            self.room = None
        self.container_dict = container_dict
        self.item_dict = item_dict
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, observation_class):
            if self.room != __value.room:
                return False
            if self.container_dict != __value.container_dict:
                return False
            if self.item_dict != __value.item_dict:
                return False

    
    def __hash__(self) -> int:
        return hash((self.room, self.container_dict, self.item_dict))
    
    def __str__(self) -> str:
        string = ""
        string += "You are at the " + self.room + ".\n"
        string += "You see the " + self.room + " is connected to the "
        step = 0
        for room in self.room_obj.connected_rooms:
            step += 1
            if step == len(self.room_obj.connected_rooms) and\
                len(self.room_obj.connected_rooms) > 1:
                string = string[:-6]+  ", and the " 
            string += room + ", the "
        string = string[:-6] + ". "
        # string += "You see "
        step = 0
        for container in self.container_dict:
            string += "The " + container + " is at the " + self.container_dict[container].position + ", "
            if step == len(self.container_dict) - 1 \
                and len(self.container_dict) > 1:
                string = string[:-2] + ", and "
            step += 1
            if container != "table":
                if self.container_dict[container].is_open:
                    string += "the " + container + " is open. "
                else:
                    string += "the " + container + " is closed. "
        for item in self.item_dict:
            if self.item_dict[item].position == IN_ROBOT_HAND:
                string += "The " + item + " is " + self.item_dict[item].position + ". "
            else:
                string += "The " + item + " is at the " + self.item_dict[item].position + ". \n"

        return string

    def update_observation(self, state):
        '''update the observation according to the state'''
        # update the room
        self.room = state.get_robot().position
        self.room_obj = state.get_room(self.room)
        # update the containers
        self.container_dict = {}
        for container in state.container_dict:
            if state.container_dict[container].position == self.room:
                self.container_dict[container] = state.container_dict[container]
        # update the items
        self.item_dict = {}
        for item in state.object_dict:
            if state.object_dict[item].position == self.room:
                self.item_dict[item] = state.object_dict[item]
            elif state.object_dict[item].position in self.container_dict:
                if self.container_dict[state.object_dict[item].position].is_open:
                    self.item_dict[item] = state.object_dict[item]
            elif state.object_dict[item].position == IN_ROBOT_HAND:
                self.item_dict[item] = state.object_dict[item]
        
    def update_model(self, state: state_class):
        '''update the state in a model according to the observation'''
        # update the room
        state.room_dict[self.room] = self.room_obj
        # update the containers, if the container is observed, its location is updated to the room
        for container in self.container_dict:
            if container not in state.object_dict:
                state.object_dict[container] = container_object(container, None, self.room, None, False, False)
            if state.object_dict[container].position != self.room:
                state.object_dict[container].position = self.room
        for container in state.object_dict:
            if container not in self.container_dict and state.object_dict[container].position == self.room:
                state.object_dict[container].position = random.choice(list(
                    copy.copy(state.room_dict.keys())).remove(self.room))

        # update the items
        for item in self.item_dict: # if object is observed but not in the state, add it to the state
            if item not in state.object_dict:
                state.object_dict[item] = item_object(name=item, 
                        position=self.room, is_moveable=False)
            if state.object_dict[item].position != self.item_dict[item].position:
                state.object_dict[item].position = self.item_dict[item].position
        for item in state.object_dict: # if object is not observed but in the state, remove it from the state
            if item not in self.item_dict:
                if state.object_dict[item] == self.item_dict[item].position:
                    state.object_dict[item].position = random.choice(list(
                        copy.copy(state.room_dict.keys())).remove(self.room) + list(
                        copy.copy(state.container_dict.keys())))
        return state

class robot_agent:
    def __init__(self, room: str, picked_item: str) -> None:
        self.position = room
        self.picked_item = picked_item

class human_agent:
    def __init__(self, room: str) -> None:
        self.position = room

class action:
    def __init__(self, container: str =None, item :str=None, room: str=None) -> None:
        self.container = container
        self.item = item    
        self.room = room

    def precondition(self, state: state_class):
        raise NotImplementedError

    def effect(self, state: state_class):
        raise NotImplementedError
    

class pick(action):
    def __init__(self, item :str=None) -> None:
        super().__init__(None, item, None)
        if item is None:
            raise ValueError("Item must be specified")
        self.string = "pick the " + item

    def precondition(self, state: state_class):
        if state.object_dict[self.item].position == state.get_robot().position:
            return True
        elif state.object_dict[self.item].position in state.container_dict:
            obj_position = state.object_dict[self.item].position
            if state.container_dict[obj_position].position == state.get_robot().position \
                and state.container_dict[obj_position].is_open:
                return True
        elif state.object_dict[self.item].position in state.surface_dict:
            obj_position = state.object_dict[self.item].position
            if state.surface_dict[obj_position].position == state.get_robot().position:
                return True
        return False

    def effect(self, state: state_class):
        if self.precondition(state) == False:
            return [state], [1.0]
        else:
            next_state = state.copy()
            next_state.object_dict[self.item].position = IN_ROBOT_HAND
            return [state, next_state], [0.1, 0.9]
    
class place(action):
    def __init__(self, item=None, position=None) -> None:
        super().__init__(None, item, None)
        self.position = position
        if item is None:
            raise ValueError("Item must be specified")
        self.string = "place the " + item + " at the " + position

    def precondition(self, state: state_class):
        if state.object_dict[self.item].position == IN_ROBOT_HAND:
            if self.position in state.room_dict \
                and self.position == state.get_robot().position:
                return True
            elif self.position in state.container_dict:
                if state.container_dict[self.position].is_open and \
                    state.container_dict[self.position].position == state.get_robot().position:
                    return True
            elif self.position in state.surface_dict:
                if state.surface_dict[self.position].position == state.get_robot().position:
                    return True
        return False

    def effect(self, state: state_class):
        if self.precondition(state) == False:
            return [state], [1.0]
        else:
            next_state = state.copy()
            next_state.object_dict[self.item].position = self.position
            return [state, next_state], [0.1, 0.9]

class move(action):
    # move to a room if it is unlocked
    def __init__(self, room=None) -> None:
        super().__init__(None, None, room)
        if room is None:
            raise ValueError("Room must be specified")
        self.string = "move to the " + room

    def precondition(self, state: state_class):
        curr_room = state.robot_agent.position
        if self.room in state.room_dict[curr_room].connected_rooms:
            return True
        return False

    def effect(self, state: state_class):
        if self.precondition(state) == False:
            return [state], [1.0]
        else:
            next_state = state.copy()
            next_state.robot_agent.position = self.room
            return [state, next_state], [0.1, 0.9]

class open(action):
    # open a container
    def __init__(self, container=None) -> None:
        super().__init__(container, None, None)
        if container is None:
            raise ValueError("Container must be specified")
        self.string = "open the " + container

    def precondition(self, state: state_class):

        if self.container is not None:
            if state.container_dict[self.container].position != state.robot_agent.position:
                return False
            if state.container_dict[self.container].is_open == True:
                return False
            elif state.container_dict[self.container].is_locked == True:
                return False 
        return True

    def effect(self, state: state_class):
        if self.precondition(state) == False:
            return [state], [1.0]
        elif self.container is not None:
            next_state = state.copy()
            next_state.container_dict[self.container].is_open = True
            return [state, next_state], [0.1, 0.9]
        else:
            return [state], [1.0]


class close(action):
    # close a container
    def __init__(self, container=None) -> None:
        super().__init__(container, None, None)
        if container is None:
            raise ValueError("Container must be specified")
        self.string = "close the " + container

    def precondition(self, state: state_class) -> bool:
        if self.container is not None:
            if state.container_dict[self.container].is_open == False:
                return False
            if state.container_dict[self.container].position \
                != state.robot_agent.position:
                return False
        return True

    def effect(self, state: state_class) -> state_class:
        if self.precondition(state) == False:
            return [state], [1.0]
        elif self.container is not None:
            next_state = state.copy()
            next_state.container_dict[self.container].is_open = False
            return [state, next_state], [0.2, 0.8]
        else: 
            return [state], [1.0]

class unlock(action):
    # unlock a room or a container
    def __init__(self, container=None, item=None, room=None) -> None:
        super().__init__(container, item, room)
        if container is None and room is None:
            raise ValueError("Either container or room must be specified")
        if container is not None and room is not None:
            raise ValueError("Only one of container or room can be specified")
        self.string = "unlock the " + item
    
    def precondition(self, state: state_class) -> bool:
        if self.container is not None:
            if self.container.key.in_robot_hand == False:
                return False
            if state.container_dict[self.container].is_locked == False:
                return False
        elif self.room is not None:
            if self.room.key.in_robot_hand == False:
                return False
            if state.room_dict[self.room].is_locked == False:
                return False
        return True
    
    def effect(self, state: state_class) -> state_class:
        if self.precondition(state) == False:
            return [state], [1.0]
        elif self.container is not None:
            next_state = state.copy()
            next_state.container_dict[self.container].is_locked = False
            return [state, next_state], [0.2, 0.8] 
        elif self.room is not None:
            next_state = state.copy()
            next_state.room_dict[self.room].is_locked = False
            return [state, next_state], [0.2, 0.8]
        else:
            return [state], [1.0]

class minihousemodel:
    def __init__(self, instruction, goal_state, verbose) -> None:
        self.state = None
        self.goal_state = goal_state
        self.instruction = instruction
        self.history = []
        self.observation = observation_class(None, None, None)
        self.verbose = verbose
    def get_valid_actions(self):
        valid_action = {}
        for action in self.action_space:
            if self.action_dict[action].precondition(self.state):
                valid_action[action] = self.action_dict[action]
        return valid_action

    def grounding_actions(self):
        raise NotImplementedError 

    def create_initial_state(self):
        raise NotImplementedError

    @staticmethod
    def goal_achieved(goal_state, state) -> bool:
        for obj in goal_state.object_dict:
            if goal_state.object_dict[obj].position != state.object_dict[obj].position:
                return False 
        return True   
    
    def is_terminal(self) -> bool:
        return self.goal_achieved(self.goal_state, self.state)  
    
    @property
    def get_state(self) -> state_class:
        return self.state 
    
    def state_to_index(self, state: state_class) -> int:
        raise NotImplementedError
    
    def index_to_state(self, index: int) -> state_class:
        raise NotImplementedError

    def index_to_action(self, index: int) -> action:
        raise NotImplementedError
    
    def action_to_index(self, action: action) -> int:
        raise NotImplementedError

    def get_reward(self) -> int:
        if self.is_terminal():
            return 50
        else:
            return -1

    def get_observation(self) -> observation_class:
        self.observation.update_observation(self.state) 
        # print(self.observation)
        return self.observation

    def transition(self, state, action) -> Tuple[observation_class, int, bool, list]:
        return_list = []
        if isinstance(state, int):
            state = self.index_to_state(state)
        if isinstance(action, int):
            action = self.index_to_action(action)
        next_state_list, prob_list = action.effect(state)
        for next_state, next_prob in zip(next_state_list, prob_list):
            done = self.goal_achieved(self.goal_state, next_state)
            reward = 50 if done else -1
            return_list.append((next_prob, self.state_to_index(next_state), reward, done)) 
        return return_list

    def step(self, action) -> Tuple[observation_class, int, bool]:
        if isinstance(action, int):
            action = self.index_to_action(action)
        if self.verbose:
            print("action: ", action.string)
        next_state_list, next_prob_list = action.effect(self.state) 
        next_state = np.random.choice(next_state_list, p=next_prob_list)
        self.state = next_state
        reward = self.get_reward()
        obs = self.get_observation()
        done = self.is_terminal()
        self.history.append(action.string)
        valid_action = self.get_valid_actions()
        return obs, reward, done, self.history, valid_action

    def copy_env(self):
        return copy.deepcopy(self)