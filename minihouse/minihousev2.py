

from minihouse.minihousemodel import item_object, \
    container_object, room_class, state_class, minihousemodel, key_object,\
    robot_agent, human_agent, move, open, close, pick, place, observation_class
ROOM_LIST = [
    "living room",
    "kitchen",
    "bedroom1",
    "bathroom1",
    "hallway",
    "bedroom2",
    "bathroom2",
    "diningroom",
    "studyroom",
    "garage",
]

CONNECTED_ROOM = {
    "hallway": ["diningroom", "bedroom1", "bedroom2", 
                "bathroom1", "living room", "studyroom", "garage"],
    "diningroom": ["hallway", "kitchen"],
    "kitchen" : ["dinigroom"],
    "bedroom": ["hallway"],
    "bathroom": ["hallway"],
    "living room": ["hallway"],
    "bedroom1": ["hallway"],
    "bathroom1": ["hallway"],
    "bedroom2": ["hallway", "bathroom2"],
    "bathroom2": ["bedroom2"],
    "studyroom": ["hallway"],
    "garage": ["hallway"],
}

ACTION_DICT = {
    "move": move,
    "open": open,
    "close": close,
    "pick": pick,
    "place": place,
}



MOVEABLE_OBJECT_LIST = [
    # from kitchen
    "apple", "banana", "orange", "milk", "plate", "bowl",
    # from bedroom1
    "white t-shirt", "green pants", "black socks", "white shoes",
    "black pants", "green t-shirt", "blue shoes", "black shoes",
    # from bathroom1
    "toothbrush", "toothpaste", "soap", "shampoo", "towel",
    # from bedroom2
    "red t-shirt", "blue pants", "white socks", "blue shorts",
    # from bathroom2
    "toothbrush2", "toothpaste2", "soap2", "shampoo2", "towel2", 
    # from livingroom
    "remote", "book", "magazine", "newspaper",
    # from diningroom
    "fork", "knife", "spoon", "tissue", 
    # from studyroom
    "pen", "pencil", "paper", "notebook",
    # from garage
    "bike", "toolbox",
]

MOVEABLE_OBJECT_DICT = {
    "apple": "fridge", # in kitchen
    "banana": "fridge", # in kitchen
    "orange": "fridge", # in kitchen
    "milk": "fridge", # in kitchen
    "white t-shirt": "closet1", # in bedroom1
    "green pants": "closet1", # in bedroom1
    "black socks": "closet1", # in bedroom1
    "white shoes": "closet1", # in bedroom1
    "black pants": "closet1", # in bedroom1
    "green t-shirt": "closet1", # in bedroom1
    "blue shoes": "closet1", # in bedroom1
    "black shoes": "closet1", # in bedroom1
    "red t-shirt": "closet2", # in bedroom2
    "blue pants": "closet2", # in bedroom2
    "white socks": "closet2", # in bedroom2
    "blue shorts": "closet2", # in bedroom2
    "remote": "table1", # in livingroom
    "book": "table1", # in livingroom
    "magazine": "table1", # in livingroom
    "newspaper": "table1", # in livingroom
    "fork": "diningtable", # in diningroom
    "knife": "diningtable", # in diningroom
    "spoon": "diningtable", # in diningroom
    "tissue": "diningtable", # in diningroom
    "plate": "shelf4", # in kitchen
    "bowl": "shelf4", # in kitchen
    "pen": "desk", # in studyroom
    "pencil": "desk", # in studyroom
    "paper": "desk", # in studyroom
    "notebook": "desk", # in studyroom
    "bike": "garage", # in garage
    "toolbox": "garage", # in garage
    "box": "basement", # in basement
    "bag": "basement", # in basement
    "basket": "basement", # in basement
    "bucket": "basement", # in basement
    "toothbrush": "shelf5", # in bathroom1
    "toothpaste": "shelf5", # in bathroom1
    "soap": "shelf5", # in bathroom1
    "shampoo": "shelf5", # in bathroom1
    "towel": "bathroom2", # in bathroom1
    "toothbrush2": "shelf6", # in bathroom2
    "toothpaste2": "shelf6", # in bathroom2
    "soap2": "shelf6", # in bathroom2
    "shampoo2": "shelf6", # in bathroom2
    "towel2": "bathroom2", # in bathroom2

}

CONTAINER_LIST = [
    "fridge", # in kitchen
    "shelf4", # in kitchen
    "dresser1", # in bedroom1
    "dresser2", # in bedroom2
    "table1", # in livingroom
    "diningtable", # in diningroom
    "desk", # in studyroom
    "closet1", # in bedroom1
    "closet2", # in bedroom2
    "shelf1", # in livingroom
    "shelf2", # in diningroom
    "shelf3", # in studyroom
    "microwave", # in kitchen
    "oven", # in kitchen
    "bed1", # in bedroom1
    "bed2", # in bedroom2
    "sofa", # in livingroom
    "chair1", # in livingroom
    "chair2", # in diningroom
    "chair3", # in studyroom
    "toilet1", # in bathroom1
    "toilet2", # in bathroom2
    "sink1", # in bathroom1
    "sink2", # in bathroom2
    "shelf5", # bathroom1
    "shelf6", # bathroom2
]

CONTAINER_DICT = {
    "fridge": "kitchen", # in kitchen
    "shelf4": "kitchen", # in kitchen
    "dresser1": "bedroom1", # in bedroom1
    "dresser2": "bedroom2", # in bedroom2
    "table1": "living room", # in livingroom
    "diningtable": "diningroom", # in diningroom
    "desk": "studyroom", # in studyroom
    "closet1": "bedroom1", # in bedroom1
    "closet2": "bedroom2", # in bedroom2
    "shelf1": "living room", # in livingroom
    "shelf2": "diningroom", # in diningroom
    "shelf3": "studyroom", # in studyroom
    "microwave": "kitchen", # in kitchen
    "oven": "kitchen", # in kitchen
    "bed1": "bedroom1", # in bedroom1
    "bed2": "bedroom2", # in bedroom2
    "sofa": "living room", # in livingroom
    "chair1": "living room", # in livingroom
    "chair2": "diningroom", # in diningroom
    "chair3": "studyroom", # in studyroom
    "toilet1": "bathroom1", # in bathroom1
    "toilet2": "bathroom2", # in bathroom2
    "sink1": "bathroom1", # in bathroom1
    "sink2": "bathroom2", # in bathroom2
    "shelf5": "bathroom1", # bathroom1
    "shelf6": "bathroom2", # bathroom2
}



class MiniHouseV2(minihousemodel):
    def __init__(self, instruction, goal_state) -> None:
        super().__init__(instruction, goal_state)
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
        room_dict = {}
        for room in ROOM_LIST:
            room_dict[room] = room_class(
                name = room,
                is_open = True,
                is_locked = False,
                key = None,
                connected_rooms=CONNECTED_ROOM[room],
            )
        for container in CONTAINER_DICT:
            if container == 'table':
                container_dict[container] = container_object(
                    name = container,
                    position = CONTAINER_DICT[container],
                    is_open = True,
                    is_locked = False,
                    key = None,
                )
            else:
                container_dict[container] = container_object(
                    name = container,
                    position = CONTAINER_DICT[container],
                    is_open = False,
                    is_locked = False,
                    key = None,
                )
        for item in MOVEABLE_OBJECT_DICT:
            item_dict[item] = item_object(
                name = item,
                position = MOVEABLE_OBJECT_DICT[item],
                is_moveable = True,
            )
        self.state = state_class(
            robot_agent = robot_agent(
                room = "living room",
                picked_item=None,
            ),
            human_agent = human_agent(
                room = "living room",
            ),
            room_list = room_dict,
            container_list = container_dict,
            object_list= item_dict,
        )

    def state_update(self, observation: observation_class):
        self.state = observation.update(self.state)

    def grounding_actions(self):
        action_dict = {}
        for action in ACTION_DICT:
            if action == "move":
                for room in ROOM_LIST:
                    action_dict[action + " to the " + room] = ACTION_DICT[action](room)
            elif action == "open":
                for container in CONTAINER_LIST:
                    action_dict[action + " the " + container] = \
                        ACTION_DICT[action](container)
            elif action == "close":
                for container in CONTAINER_LIST:
                    action_dict[action +  " the " + container] = \
                        ACTION_DICT[action](container)
            elif action == "pick":
                for item in MOVEABLE_OBJECT_DICT:
                    action_dict[action +  " the " + item] = ACTION_DICT[action](item)
            elif action == "place":
                for item in MOVEABLE_OBJECT_DICT:
                    for container in CONTAINER_LIST:
                        action_dict[action +  " the " + item + " at the " \
                            +  container] = ACTION_DICT[action](item, container)
                    for room in ROOM_LIST:
                        action_dict[action + " the " +  item + " at the " +\
                            room] = ACTION_DICT[action](item, room)
        self.action_dict = action_dict
        self.action_space = list(action_dict.keys())
        self.action_num = len(self.action_space)