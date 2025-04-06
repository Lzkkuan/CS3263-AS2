from minihouse.minihousemodel import state_class, item_object, move, open, close, pick, place
def test_case_0 ():
    instruction = "move the apple to the table"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"apple": item_object("apple", True, "table")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
    ]

    OBJECT_LIST = [
        "apple",
        # "t-shirt",
        # "cup"
    ]

    OBJECT_POSITION_LIST = {
        "apple": "fridge",
        # "t-shirt": "closet",
        # "cup": "table",
    }

    CONTAINER_LIST = [
        "fridge",
        # "closet",
        # "shelf",
    ]

    SURFACE_LIST = [
        "table",
    ]

    CONTAINER_POSITION_LIST = {
        "fridge": "kitchen",
        # "closet": "bedroom",
        "table": "living room",
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom"],
    }

    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        "open fridge",
        "close fridge",
        "pick apple",
        "place apple fridge",
        "place apple table",
        "place apple bedroom",
        "place apple bathroom",
        "place apple living room",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
    ]


    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)

def test_case_1 ():
    instruction = "move the t-shirt to the washing machine"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"t-shirt": item_object("t-shirt", True, "washing machine")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
    ]

    OBJECT_LIST = [
        # "apple",
        "t-shirt",
        # "cup"
    ]

    OBJECT_POSITION_LIST = {
        # "apple": "fridge",
        "t-shirt": "closet",
        # "cup": "table",
    }

    CONTAINER_LIST = [
        # "fridge",
        "closet",
        "washing machine"
        # "shelf",
    ]

    SURFACE_LIST = [
        # "table",
    ]

    CONTAINER_POSITION_LIST = {
        # "fridge": "kitchen",
        "closet": "bedroom",
        # "table": "living room",
        "washing machine": "kitchen",
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom"],
    }


    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        # "open fridge",
        # "close fridge",
        "open washing machine",
        "close washing machine",
        "pick t-shirt",
        "place t-shirt fridge",
        "place t-shirt table",
        "place t-shirt bedroom",
        "place t-shirt bathroom",
        "place t-shirt living room",
        "place t-shirt washing machine",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
    ]

    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)

def test_case_2 ():
    instruction = "move the drink to the desk"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"drink": item_object("drink", True, "desk")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
    ]

    OBJECT_LIST = [
        # "apple",
        # "t-shirt",
        "drink"
    ]

    OBJECT_POSITION_LIST = {
        # "apple": "fridge",
        # "t-shirt": "closet",
        "drink": "fridge",
    }

    CONTAINER_LIST = [
        "fridge",
        "closet",
        # "washing machine"
        # "shelf",
    ]

    SURFACE_LIST = [
        "table",
        "desk",
    ]

    CONTAINER_POSITION_LIST = {
        "fridge": "kitchen",
        "closet": "bedroom",
        "table": "living room",
        # "washing machine": "kitchen",
        "desk": "bedroom"
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom"],
    }


    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        "open fridge",
        "close fridge",
        # "open washing machine",
        # "close washing machine",
        "pick drink",
        # "pick apple",
        "place drink fridge",
        "place drink table",
        "place drink desk",
        "place drink bedroom",
        "place drink bathroom",
        "place drink living room",
        # "place drink washing machine",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
    ]

    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)


def test_case_3 ():
    instruction = "move the broom to the kitchen"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"broom": item_object("broom", True, "kitchen")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
        "store room"
    ]

    OBJECT_LIST = [
        # "apple",
        # "t-shirt",
        # "cup",
        "broom"
    ]

    OBJECT_POSITION_LIST = {
        # "apple": "fridge",
        # "t-shirt": "closet",
        # "cup": "table",
        "broom": "store room"
    }

    CONTAINER_LIST = [
        "fridge",
        "closet",
        # "washing machine"
        # "shelf",
    ]

    SURFACE_LIST = [
        # "table",
    ]

    CONTAINER_POSITION_LIST = {
        "fridge": "kitchen",
        "closet": "bedroom",
        # "table": "living room",
        # "washing machine": "kitchen",
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom", "store room"],
        "store room": ["living room"]
    }


    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        "open fridge",
        "close fridge",
        "open washing machine",
        "close washing machine",
        "pick broom",
        "place broom fridge",
        "place broom table",
        "place broom bedroom",
        "place broom bathroom",
        "place broom living room",
        "place broom washing machine",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
        "move store room",

    ]

    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)

def test_case_4 ():
    instruction = "move the soap to the bathroom"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"soap": item_object("soap", True, "bathroom")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
        "store room"
    ]

    OBJECT_LIST = [
        # "apple",
        # "t-shirt",
        # "cup",
        # "broom",
        "soap"
    ]

    OBJECT_POSITION_LIST = {
        # "apple": "fridge",
        # "t-shirt": "closet",
        # "cup": "table",
        # "broom": "store room",
        "soap": "cabinet",
    }

    CONTAINER_LIST = [
        # "fridge",
        "closet",
        "cabinet"
        # "washing machine"
        # "shelf",
    ]

    SURFACE_LIST = [
        # "table",
    ]

    CONTAINER_POSITION_LIST = {
        # "fridge": "kitchen",
        "closet": "bedroom",
        "cabinet": "store room"
        # "table": "living room",
        # "washing machine": "kitchen",
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom", "store room"],
        "store room": ["living room"]
    }


    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        # "open fridge",
        # "close fridge",
        # "open washing machine",
        # "close washing machine",
        "pick soap",
        "place soap fridge",
        "place soap table",
        "place soap bedroom",
        "place soap bathroom",
        "place soap living room",
        "place soap washing machine",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
        "move store room",

    ]

    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)

def test_case_5 ():
    instruction = "move the pot to the cabinet"

    goal_state = state_class(
        robot_agent=None,
        human_agent=None,
        object_list={"pot": item_object("pot", True, "cabinet")},
        container_list=None,
        surface_list=None,
        room_list=None,
    )

    ROOM_LIST = [
        "living room",
        "kitchen",
        "bedroom",
        "bathroom",
        "store room"
    ]

    OBJECT_LIST = [
        # "apple",
        # "t-shirt",
        # "cup",
        # "broom",
        "pot"
    ]

    OBJECT_POSITION_LIST = {
        # "apple": "fridge",
        # "t-shirt": "closet",
        # "cup": "table",
        # "broom": "store room",
        "pot": "under sink",
    }

    CONTAINER_LIST = [
        # "fridge",
        # "closet",
        "cabinet",
        "under sink",
        # "washing machine"
        # "shelf",
    ]

    SURFACE_LIST = [
        # "table",
    ]

    CONTAINER_POSITION_LIST = {
        # "fridge": "kitchen",
        # "closet": "bedroom",
        "cabinet": "store room",
        "under sink": "kitchen"
        # "table": "living room",
        # "washing machine": "kitchen",
        # "shelf": "living room",
    }

    CONNECTED_ROOM = {
        "kitchen": ["living room"],
        "bedroom": ["bathroom", "living room"],
        "bathroom": ["bedroom"],
        "living room": ["kitchen", "bedroom", "store room"],
        "store room": ["living room"]
    }


    ACTION_DICT = {
        "move": move,
        "open": open,
        "close": close,
        "pick": pick,
        "place": place,
    }

    GROUNDED_ACTION_LIST = [
        # "open fridge",
        # "close fridge",
        # "open washing machine",
        # "close washing machine",
        "pick pot",
        # "place pot fridge",
        # "place pot table",
        # "place pot bedroom",
        # "place pot bathroom",
        # "place pot living room",
        # "place pot washing machine",
        "place pot store room",
        "place pot cabinet",
        "move living room",
        "move kitchen",
        "move bedroom",
        "move bathroom",
        "move store room",

    ]

    return instruction, goal_state, (ROOM_LIST, OBJECT_LIST, OBJECT_POSITION_LIST, \
    CONTAINER_LIST, SURFACE_LIST, CONTAINER_POSITION_LIST, CONNECTED_ROOM, \
    ACTION_DICT, GROUNDED_ACTION_LIST)

test_cases = [test_case_0(), test_case_1(), test_case_2(), test_case_3(), test_case_4(), test_case_5()]

# state sizes: [(56,13), (111,15), (114,17), (170,17), (160, 17), (160, 17)]
