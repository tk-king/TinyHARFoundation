from whar_datasets import WHARDatasetID

# Original: ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
CLASS_NAMES_UCI_HAR = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]

# Original: ['Walking', 'Jogging', 'Upstairs', 'Downstairs', 'Sitting', 'Standing']
#CLASS_NAMES_WISDM = ["walking", "running", "jogging", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
CLASS_NAMES_WISDM = ["walking", "jogging", "walking_upstairs", "walking_downstairs", "sitting", "standing"]

# Original: "downstairs", "upstairs", "walking", "jogging", "sitting", "standing",
CLASS_NAMES_MOTIONSENSE = ["walking_downstairs", "walking_upstairs", "walking", "jogging", "sitting", "standing"]


# whar_datasets OPPORTUNITY uses the 18-class gesture recognition task.
# Source: `get_dataset_cfg(WHARDatasetID.OPPORTUNITY).activity_names`
# CLASS_NAMES_OPPORTUNITY = [
#     "unknown",
#     "open_door_1",
#     "open_door_2",
#     "close_door_1",
#     "close_door_2",
#     "open_fridge",
#     "close_fridge",
#     "open_dishwasher",
#     "close_dishwasher",
#     "open_drawer_1",
#     "close_drawer_1",
#     "open_drawer_2",
#     "close_drawer_2",
#     "open_drawer_3",
#     "close_drawer_3",
#     "clean_table",
#     "drink_from_cup",
#     "toggle_switch",
# ]


CLASS_NAMES_OPPORTUNITY = [
    "unknown",
    "open_door",
    "open_door",
    "close_door",
    "close_door",
    "open_fridge",
    "close_fridge",
    "open_dishwasher",
    "close_dishwasher",
    "open_drawer",
    "close_drawer",
    "open_drawer",
    "close_drawer",
    "open_drawer",
    "close_drawer",
    "clean_table",
    "drinking",
    "toggle_switch",
]


# Original: "lying", "sitting", "standing", "walking", "running", "cycling", "nordic walking", "watching TV", "computer work", "car driving", "ascending stairs", "descending stairs", "vacuum cleaning", "ironing", "folding laundry", "house cleaning", "playing soccer", "rope jumping",
CLASS_NAMES_PAMAP2 = ["laying", "sitting", "standing", "walking", "running", "cycling", "nordic_walking", "watching_tv", "computer_work", "car_driving", "walking_upstairs", "walking_downstairs", "vacuum_cleaning", "ironing", "folding_laundry", "house_cleaning", "playing_soccer", "rope_jumping"]

# Original: "Standing still", "Sitting and relaxing", "Lying down", "Walking", "Climbing stairs", "Waist bends forward", "Frontal elevation of arms", "Knees bending (crouching)", "Cycling", "Jogging", "Running", "Jump front and back",
CLASS_NAMES_MHEALTH = ["null", "standing", "sitting", "laying", "walking", "walking_upstairs", "waist_bends_forward", "frontal_elevation_of_arms", "knees_bending_crouching", "cycling", "jogging", "running", "jump_front_and_back"]

# Original: "sitting", "standing", "lying on back", "lying on right side", "ascending stairs", "descending stairs", "standing in an elevator still", "moving around in an elevator", "walking in a parking lot", "walking on treadmill (flat, 4 km/h)", "walking on treadmill (15° incline, 4 km/h)", "running on treadmill (8 km/h)", "exercising on a stepper", "exercising on a cross trainer", "cycling on exercise bike (horizontal)", "cycling on exercise bike (vertical)", "rowing", "jumping", "playing basketball"
CLASS_NAMES_DSADS = ["sitting", "standing", "laying", "laying", "walking_upstairs", "walking_downstairs", "standing_in_an_elevator_still", "moving_around_in_an_elevator", "walking", "walking_threadmill", "walking_threadmill", "running_threadmill", "exercise_stepper", "exercise_cross_trainer", "cycling_on_exercise_bike_horizontal", "cycling_on_exercise_bike_vertical", "rowing", "jumping", "playing_basketball"]

# TODO: FIX
# Stand", "Sit", "Talk-sit", "Talk-stand", "Stand-sit", "Lay", "Lay-stand", "Pick", "Jump", "Push-up", "Sit-up", "Walk", "Walk-backward", "Walk-circle", "Run", "Stair-up", "Stair-down", "Table-tennis"
CLASS_NAMES_KU_HAR = ["standing", "sitting", "talking_sitting", "talking_standing", "Stand-sit", "laying", "laying_standing", "picking", "jumping", "push_up", "sit_up", "walking", "walking_backward", "walking_circle", "running", "stair_up", "stair_down", "table_tennis"]

# Original: "Walking", "Standing", "Upstairs", "Downstairs", "Running", "Sitting"
CLASS_NAMES_HAR_SENSE = ["walking", "standing", "walking_upstairs", "walking_downstairs", "running", "sitting"]

class_names = {
    WHARDatasetID.UCI_HAR: CLASS_NAMES_UCI_HAR,
    #WHARDatasetID.WISDM: CLASS_NAMES_WISDM,
    WHARDatasetID.MOTION_SENSE: CLASS_NAMES_MOTIONSENSE,
    WHARDatasetID.OPPORTUNITY: CLASS_NAMES_OPPORTUNITY,
    WHARDatasetID.PAMAP2: CLASS_NAMES_PAMAP2,
    WHARDatasetID.MHEALTH: CLASS_NAMES_MHEALTH,
    WHARDatasetID.DSADS: CLASS_NAMES_DSADS,
    WHARDatasetID.KU_HAR: CLASS_NAMES_KU_HAR,
    WHARDatasetID.HAR_SENSE: CLASS_NAMES_HAR_SENSE,
}

def get_class_names(dataset_id: WHARDatasetID):
    return class_names[dataset_id]


all_class_names = set()
for names in class_names.values():
    all_class_names.update(names)
all_class_names = sorted(all_class_names)
global_class_map = {name: idx for idx, name in enumerate(all_class_names)}

print("++++++ Global class name mapping:")
for name, idx in global_class_map.items():
    print(f"{idx}: {name}")


def get_global_class_name_map(dataset_id: WHARDatasetID):
    local_names = get_class_names(dataset_id)
    name_to_global_index = {name: global_class_map[name] for name in local_names}
    return name_to_global_index



def sanitize_class_names(class_names):
    sanitization_map = {
        "null": "Doing nothing",
        "walking": "Walking",
        "jogging": "Jogging",
        "standing": "Standing",
        "sitting": "Sitting",
        "laying": "Laying",
        "running": "Running",
        "cycling": "Cycling",
        "nordic_walking": "Nordic Walking",
        "watching_tv": "Watching TV",
        "computer_work": "Computer Work",
        "car_driving": "Car Driving",
        "walking_upstairs": "Walking Upstairs",
        "walking_downstairs": "Walking Downstairs",
        "walking_threadmill": "Walking",
        "vacuum_cleaning": "Vacuum Cleaning",
        "ironing": "Ironing",
        "folding_laundry": "Folding Laundry",
        "house_cleaning": "House Cleaning",
        "playing_soccer": "Playing Soccer",
        "playing_basketball": "Playing Basketball",
        "rope_jumping": "Rope Jumping",
        "waist_bends_forward": "Bending forward using the waist",
        "frontal_elevation_of_arms": "Elevating the arms in the front",
        "knees_bending_crouching": "Bending the knees and crouching",
        "jump_front_and_back": "Jumping front and back",
        "jumping": "Jumping",
        "running_threadmill": "Running",
        "cycling_on_exercise_bike_horizontal": "Cycling",
        "cycling_on_exercise_bike_vertical": "Cycling",
        "exercise_stepper": "Stepper Exercise",
        "exercise_cross_trainer": "Cross Trainer",
        "rowing": "Rowing",
        "standing_in_an_elevator_still": "Standing in an Elevator (Still)",
        "moving_around_in_an_elevator": "Moving Around in an Elevator",
        "walking_backward": "Walking Backward",
        "walking_circle": "Walking in a Circle",
        "picking": "Picking",
        "push_up": "Push-up",
        "sit_up": "Sit-up",
        "talking_sitting": "Talking while sitting",
        "talking_standing": "Talking while standing",
        "laying_standing": "Laying to standing",
        "table_tennis": "Table Tennis",
    }
    sanitized = []
    for name in class_names:
        pretty = sanitization_map.get(name.lower())
        if pretty is None:
            pretty = name.replace("_", " ").strip()
            if pretty.islower():
                pretty = pretty.title()
        sanitized.append(pretty)
    return sanitized
