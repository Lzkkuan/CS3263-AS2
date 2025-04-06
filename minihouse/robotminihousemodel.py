from minihouse.minihousemodel import minihousemodel, state_class, robot_agent, human_agent, \
    item_object, surface_object, container_object, room_class, move, open, \
    close, pick, place, observation_class, key_object

class MiniHouseV1(minihousemodel):
    def __init__(self, instruction, goal_state, initial_conditions, verbose) -> None:
        super().__init__(instruction, goal_state, verbose)
        self.ROOM_LIST, self.OBJECT_LIST, self.OBJECT_POSITION_LIST, \
            self.CONTAINER_LIST, self.SURFACE_LIST, self.CONTAINER_POSITION_LIST, self.CONNECTED_ROOM, \
            self.ACTION_DICT, self.GROUNDED_ACTION_LIST = initial_conditions
        self.create_initial_state()
        self.grounding_actions()
    def reset(self):
        self.create_initial_state()
        reward = self.get_reward()
        obs = self.get_observation()
        done = self.is_terminal()
        valid_action = self.get_valid_actions()
        return obs, reward, done, self.history, valid_action

    def create_initial_state(self):
        item_dict = {}
        container_dict = {}
        surface_dict = {}
        room_dict = {}
        for room in self.ROOM_LIST:
            room_dict[room] = room_class(
                name = room,
                is_open = True,
                is_locked = False,
                key = None,
                connected_rooms=self.CONNECTED_ROOM[room],
            )
        for container in self.CONTAINER_LIST:
            container_dict[container] = container_object(
                name = container,
                position = self.CONTAINER_POSITION_LIST[container],
                is_open = False,
                is_locked = False,
                key = None,
            )
        for surface in self.SURFACE_LIST:
            surface_dict[surface] = surface_object(
                name = surface,
                position = self.CONTAINER_POSITION_LIST[surface],
            )
        for item in self.OBJECT_LIST:
            item_dict[item] = item_object(
                name = item,
                position = self.OBJECT_POSITION_LIST[item],
                is_moveable = True,
            )
        self.state = state_class(
            robot_agent = robot_agent(
                room = "kitchen",
                picked_item=None,
            ),
            human_agent = human_agent(
                room = "living room",
            ),
            room_list = room_dict,
            container_list = container_dict,
            surface_list=surface_dict,
            object_list= item_dict,
        )

    def state_update(self, observation: observation_class):
        self.state = observation.update(self.state)

    @property
    def nS(self):
        num_of_states = 1
        for item in self.OBJECT_LIST:
            num_of_states *= (len(self.CONTAINER_LIST) + len(self.SURFACE_LIST) + len(self.ROOM_LIST) + 1)
        for container in self.CONTAINER_LIST:
            num_of_states *= 2
        # num_of_states = num_of_states / (len(CONTAINER_LIST) + len(SURFACE_LIST) + len(ROOM_LIST)) * (len(CONTAINER_LIST) + len(SURFACE_LIST) + len(ROOM_LIST) + 1)
        num_of_states *= len(self.ROOM_LIST)
        return int(num_of_states)

    @property 
    def nA(self):
        return self.action_num
    
    def action_to_index(self, action) -> int:
        return self.action_space.index(action)

    def index_to_action(self, index: int) -> str:
        return self.action_dict[self.action_space[index]]

    def state_to_index(self, state: state_class) -> int:
        # state to index
        # use one int to represent a state
        # a state consists of all object's position and agent's position
        # the order of the object is fixed
        # the order of the container is fixed
        # the order of the room is fixed
        # there are robot agent and human agent
        index=0
        object_picked = False
        for item in self.OBJECT_LIST:
            index = index * (len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + len(self.ROOM_LIST) + 1)
            object_position = state.object_dict[item]
            if object_position.position in self.SURFACE_LIST:
                index += self.SURFACE_LIST.index(object_position.position)
            elif object_position.position in self.CONTAINER_LIST:
                index += len(self.SURFACE_LIST) + self.CONTAINER_LIST.index(object_position.position)
            elif object_position.position in self.ROOM_LIST:
                index += len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + self.ROOM_LIST.index(object_position.position)
            else:
                index += len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + len(self.ROOM_LIST)
            # else:
            #     raise ValueError("object position is not valid")
        for container in self.CONTAINER_LIST:
            index = index * 2
            container_item = state.container_dict[container]
            if container_item.is_open:
                index += 1
            else:
                index += 0
        index = index * (len(self.ROOM_LIST))
        robot_pos = state.robot_agent.position
        index += self.ROOM_LIST.index(robot_pos)
        return int(index)


    def index_to_state(self, index: int) -> state_class:
        # index to state
        # use one int to represent a state
        # a state consists of all object's position and agent's position
        # the order of the object is fixed
        # the order of the container is fixed
        # the order of the room is fixed
        # there is only one agent

        # get robot agent index
        robot_agent_index = index % (len(self.ROOM_LIST))
        index = index // (len(self.ROOM_LIST))
        # get container open or closed
        container_open = []
        for container in self.CONTAINER_LIST:
            container_open.append(index % 2)
            index = index // 2
        container_open.reverse()

        # get object index
        object_index = []
        for item in self.OBJECT_LIST:
            object_index.append(index % (len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + len(self.ROOM_LIST) + 1))
            index = index // (len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + len(self.ROOM_LIST) + 1)
        object_index.reverse()

        object_dict = {}
        for i in range(len(self.OBJECT_LIST)):
            item = self.OBJECT_LIST[i]
            if object_index[i] < len(self.SURFACE_LIST):
                object_dict[item] = item_object(
                    name = item,
                    position = self.SURFACE_LIST[object_index[i]],
                    is_moveable = True,
                )
            elif object_index[i] < len(self.SURFACE_LIST) + len(self.CONTAINER_LIST):
                object_dict[item] = item_object(
                    name = item,
                    position = self.CONTAINER_LIST[object_index[i] - len(self.SURFACE_LIST)],
                    is_moveable = True,
                )
            elif object_index[i] < len(self.SURFACE_LIST) + len(self.CONTAINER_LIST) + len(self.ROOM_LIST) :
                object_dict[item] = item_object(
                    name = item,
                    position = self.ROOM_LIST[object_index[i] - len(self.SURFACE_LIST) - len(self.CONTAINER_LIST)],
                    is_moveable = True,
                )
            else:
                object_dict[item] = item_object(
                    name = item,
                    position = "picked by you",
                    is_moveable = True,
                )
        # create container
        container_dict = {}
        for i in range(len(self.CONTAINER_LIST)):
            container = self.CONTAINER_LIST[i]
            container_dict[container] = container_object(
                name = container,
                position = self.CONTAINER_POSITION_LIST[container],
                is_open = container_open[i],
                is_locked = False,
                key = None,
            )
        
        # create robot agent
        robotagent = robot_agent(
            room = self.ROOM_LIST[robot_agent_index],
            picked_item=None,
        )
        # create state
        state = state_class(
            robot_agent = robotagent,
            human_agent = self.state.human_agent,
            room_list = self.state.room_dict,
            container_list = container_dict,
            surface_list=self.state.surface_dict,
            object_list= object_dict,
        )
        return state 

    def grounding_actions(self):
        action_dict = {}
        for action in self.ACTION_DICT:
            if action == "move":
                for room in self.ROOM_LIST:
                    action_dict[action + " to the " + room] = self.ACTION_DICT[action](room)
            elif action == "open":
                for container in self.CONTAINER_LIST:
                    action_dict[action + " the " + container] = \
                        self.ACTION_DICT[action](container)
            elif action == "close":
                for container in self.CONTAINER_LIST:
                    action_dict[action +  " the " + container] = \
                        self.ACTION_DICT[action](container)
            elif action == "pick":
                for item in self.OBJECT_LIST:
                    action_dict[action +  " the " + item] = self.ACTION_DICT[action](item)
            elif action == "place":
                for item in self.OBJECT_LIST:
                    for container in self.CONTAINER_LIST:
                        action_dict[action +  " the " + item + " at the " \
                            +  container] = self.ACTION_DICT[action](item, container)
                    for surface in self.SURFACE_LIST:
                        action_dict[action +  " the " + item + " at the " \
                            +  surface] = self.ACTION_DICT[action](item, surface)
                    for room in self.ROOM_LIST:
                        action_dict[action + " the " +  item + " at the " +\
                            room] = self.ACTION_DICT[action](item, room)
        self.action_dict = action_dict
        self.action_space = list(action_dict.keys())
        self.action_num = len(self.action_space)