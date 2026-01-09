from enum import Enum
from whar_datasets import WHARDatasetID

# class SensorLocation(int, Enum):
#     HIP = 0
#     HAND = 1
#     CHEST = 2
#     ANKLE = 3
#     LEFT_ANKLE = 3
#     RIGHT_ANKLE = 3
#     RIGHT_ARM = 4
#     LEFT_ARM = 4

class SensorLocation(int, Enum):
    TORSO = 0
    HIP = 1
    HEAD = 2
    LEGS = 3
    ARMS = 4


# class SensorType(Enum):
#     ACC_X = 0
#     ACC_Y = 1
#     ACC_Z = 2
#     GYRO_X = 3
#     GYRO_Y = 4
#     GYRO_Z = 5
#     MAG_X = 6
#     MAG_Y = 7
#     MAG_Z = 8
#     ECG = 9
#     BODY_ACC_X = 10
#     BODY_ACC_Y = 11
#     BODY_ACC_Z = 12
#     BODY_GYRO_X = 13
#     BODY_GYRO_Y = 14
#     BODY_GYRO_Z = 15
#     ATTITUE_ROLL = 16
#     ATTITUE_PITCH = 17
#     ATTITUE_YAW = 18
#     GRAVITY_X = 19
#     GRAVITY_Y = 20
#     GRAVITY_Z = 21
#     ROTATION_RATE_X = 22
#     ROTATION_RATE_Y = 23
#     ROTATION_RATE_Z = 24
#     USERACCLERATION_X = 25
#     USERACCLERATION_Y = 26
#     USERACCLERATION_Z = 27


class SensorType(Enum):
    ACC_X = 0
    ACC_Y = 1
    ACC_Z = 2
    GYRO_X = 3
    GYRO_Y = 4
    GYRO_Z = 5
    MAG_X = 6
    MAG_Y = 7
    MAG_Z = 8
    ECG = 9
    BODY_ACC_X = 0
    BODY_ACC_Y = 1
    BODY_ACC_Z = 2
    BODY_GYRO_X = 3
    BODY_GYRO_Y = 4
    BODY_GYRO_Z = 5
    ATTITUE_ROLL = 16
    ATTITUE_PITCH = 17
    ATTITUE_YAW = 18
    GRAVITY_X = 19
    GRAVITY_Y = 20
    GRAVITY_Z = 21
    ROTATION_RATE_X = 3
    ROTATION_RATE_Y = 4
    ROTATION_RATE_Z = 5
    USERACCLERATION_X = 0
    USERACCLERATION_Y = 1
    USERACCLERATION_Z = 2


UCI_HAR_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
    (SensorLocation.HIP, SensorType.BODY_ACC_X),
    (SensorLocation.HIP, SensorType.BODY_ACC_Y),
    (SensorLocation.HIP, SensorType.BODY_ACC_Z),
    (SensorLocation.HIP, SensorType.BODY_GYRO_X),
    (SensorLocation.HIP, SensorType.BODY_GYRO_Y),
    (SensorLocation.HIP, SensorType.BODY_GYRO_Z)
]

WIDSM_SENSOR_TYPES = [ # For WISDM, the authors name the position as "pocket"
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
]

MOTIONSENSE_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ATTITUE_ROLL),
    (SensorLocation.HIP, SensorType.ATTITUE_PITCH),
    (SensorLocation.HIP, SensorType.ATTITUE_YAW),
    (SensorLocation.HIP, SensorType.GRAVITY_X),
    (SensorLocation.HIP, SensorType.GRAVITY_Y),
    (SensorLocation.HIP, SensorType.GRAVITY_Z),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_X),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_Y),
    (SensorLocation.HIP, SensorType.ROTATION_RATE_Z),
    (SensorLocation.HIP, SensorType.USERACCLERATION_X),
    (SensorLocation.HIP, SensorType.USERACCLERATION_Y),
    (SensorLocation.HIP, SensorType.USERACCLERATION_Z),
]

PAMAP2_SENSOR_TYPES = [
    (SensorLocation.ARMS, SensorType.ACC_X),
    (SensorLocation.ARMS, SensorType.ACC_Y),
    (SensorLocation.ARMS, SensorType.ACC_Z),
    (SensorLocation.ARMS, SensorType.ACC_X),
    (SensorLocation.ARMS, SensorType.ACC_Y),
    (SensorLocation.ARMS, SensorType.ACC_Z),
    (SensorLocation.ARMS, SensorType.GYRO_X),
    (SensorLocation.ARMS, SensorType.GYRO_Y),
    (SensorLocation.ARMS, SensorType.GYRO_Z),
    (SensorLocation.ARMS, SensorType.MAG_X),
    (SensorLocation.ARMS, SensorType.MAG_Y),
    (SensorLocation.ARMS, SensorType.MAG_Z),

    (SensorLocation.TORSO, SensorType.ACC_X),
    (SensorLocation.TORSO, SensorType.ACC_Y),
    (SensorLocation.TORSO, SensorType.ACC_Z),
    (SensorLocation.TORSO, SensorType.ACC_X),
    (SensorLocation.TORSO, SensorType.ACC_Y),
    (SensorLocation.TORSO, SensorType.ACC_Z),
    (SensorLocation.TORSO, SensorType.GYRO_X),
    (SensorLocation.TORSO, SensorType.GYRO_Y),
    (SensorLocation.TORSO, SensorType.GYRO_Z),
    (SensorLocation.TORSO, SensorType.MAG_X),
    (SensorLocation.TORSO, SensorType.MAG_Y),
    (SensorLocation.TORSO, SensorType.MAG_Z),

    (SensorLocation.LEGS, SensorType.ACC_X),
    (SensorLocation.LEGS, SensorType.ACC_Y),
    (SensorLocation.LEGS, SensorType.ACC_Z),
    (SensorLocation.LEGS, SensorType.ACC_X),
    (SensorLocation.LEGS, SensorType.ACC_Y),
    (SensorLocation.LEGS, SensorType.ACC_Z),
    (SensorLocation.LEGS, SensorType.GYRO_X),
    (SensorLocation.LEGS, SensorType.GYRO_Y),
    (SensorLocation.LEGS, SensorType.GYRO_Z),
    (SensorLocation.LEGS, SensorType.MAG_X),
    (SensorLocation.LEGS, SensorType.MAG_Y),
    (SensorLocation.LEGS, SensorType.MAG_Z)
]

OPPORTUNITY_SENSOR_TYPES = [

]

MHEALTH_SENSOR_TYPES = [
    (SensorLocation.TORSO, SensorType.ACC_X),
    (SensorLocation.TORSO, SensorType.ACC_Y),
    (SensorLocation.TORSO, SensorType.ACC_Z),
    (SensorLocation.TORSO, SensorType.ECG),
    (SensorLocation.TORSO, SensorType.ECG),
    (SensorLocation.LEGS, SensorType.ACC_X),
    (SensorLocation.LEGS, SensorType.ACC_Y),
    (SensorLocation.LEGS, SensorType.ACC_Z),
    (SensorLocation.LEGS, SensorType.GYRO_X),
    (SensorLocation.LEGS, SensorType.GYRO_Y),
    (SensorLocation.LEGS, SensorType.GYRO_Z),
    (SensorLocation.LEGS, SensorType.MAG_X),
    (SensorLocation.LEGS, SensorType.MAG_Y),
    (SensorLocation.LEGS, SensorType.MAG_Z),
    (SensorLocation.ARMS, SensorType.ACC_X),
    (SensorLocation.ARMS, SensorType.ACC_Y),
    (SensorLocation.ARMS, SensorType.ACC_Z),
    (SensorLocation.ARMS, SensorType.GYRO_X),
    (SensorLocation.ARMS, SensorType.GYRO_Y),
    (SensorLocation.ARMS, SensorType.GYRO_Z),
    (SensorLocation.ARMS, SensorType.MAG_X),
    (SensorLocation.ARMS, SensorType.MAG_Y),
    (SensorLocation.ARMS, SensorType.MAG_Z)
]

DSADS_SENSOR_TYPES = [
    (SensorLocation.TORSO, SensorType.ACC_X),
    (SensorLocation.TORSO, SensorType.ACC_Y),
    (SensorLocation.TORSO, SensorType.ACC_Z),
    (SensorLocation.TORSO, SensorType.GYRO_X),
    (SensorLocation.TORSO, SensorType.GYRO_Y),
    (SensorLocation.TORSO, SensorType.GYRO_Z),
    (SensorLocation.TORSO, SensorType.MAG_X),
    (SensorLocation.TORSO, SensorType.MAG_Y),
    (SensorLocation.TORSO, SensorType.MAG_Z),

    (SensorLocation.ARMS, SensorType.ACC_X),
    (SensorLocation.ARMS, SensorType.ACC_Y),
    (SensorLocation.ARMS, SensorType.ACC_Z),
    (SensorLocation.ARMS, SensorType.GYRO_X),
    (SensorLocation.ARMS, SensorType.GYRO_Y),
    (SensorLocation.ARMS, SensorType.GYRO_Z),
    (SensorLocation.ARMS, SensorType.MAG_X),
    (SensorLocation.ARMS, SensorType.MAG_Y),
    (SensorLocation.ARMS, SensorType.MAG_Z),

    (SensorLocation.ARMS, SensorType.ACC_X),
    (SensorLocation.ARMS, SensorType.ACC_Y),
    (SensorLocation.ARMS, SensorType.ACC_Z),
    (SensorLocation.ARMS, SensorType.GYRO_X),
    (SensorLocation.ARMS, SensorType.GYRO_Y),
    (SensorLocation.ARMS, SensorType.GYRO_Z),
    (SensorLocation.ARMS, SensorType.MAG_X),
    (SensorLocation.ARMS, SensorType.MAG_Y),
    (SensorLocation.ARMS, SensorType.MAG_Z),

    (SensorLocation.LEGS, SensorType.ACC_X),
    (SensorLocation.LEGS, SensorType.ACC_Y),
    (SensorLocation.LEGS, SensorType.ACC_Z),
    (SensorLocation.LEGS, SensorType.GYRO_X),
    (SensorLocation.LEGS, SensorType.GYRO_Y),
    (SensorLocation.LEGS, SensorType.GYRO_Z),
    (SensorLocation.LEGS, SensorType.MAG_X),
    (SensorLocation.LEGS, SensorType.MAG_Y),
    (SensorLocation.LEGS, SensorType.MAG_Z),

    (SensorLocation.LEGS, SensorType.ACC_X),
    (SensorLocation.LEGS, SensorType.ACC_Y),
    (SensorLocation.LEGS, SensorType.ACC_Z),
    (SensorLocation.LEGS, SensorType.GYRO_X),
    (SensorLocation.LEGS, SensorType.GYRO_Y),
    (SensorLocation.LEGS, SensorType.GYRO_Z),
    (SensorLocation.LEGS, SensorType.MAG_X),
    (SensorLocation.LEGS, SensorType.MAG_Y),
    (SensorLocation.LEGS, SensorType.MAG_Z)
]

KU_HAR_SENSOR_TYPES = [
    (SensorLocation.HIP, SensorType.ACC_X),
    (SensorLocation.HIP, SensorType.ACC_Y),
    (SensorLocation.HIP, SensorType.ACC_Z),
    (SensorLocation.HIP, SensorType.GYRO_X),
    (SensorLocation.HIP, SensorType.GYRO_Y),
    (SensorLocation.HIP, SensorType.GYRO_Z),
]

HAR_SENSE_SENSOR_TYPES = [

]

datasets_to_types = {
    WHARDatasetID.UCI_HAR: UCI_HAR_SENSOR_TYPES,
    #WHARDatasetID.WISDM: WIDSM_SENSOR_TYPES,
    WHARDatasetID.MOTION_SENSE: MOTIONSENSE_SENSOR_TYPES,
    WHARDatasetID.PAMAP2: PAMAP2_SENSOR_TYPES,
    WHARDatasetID.OPPORTUNITY: OPPORTUNITY_SENSOR_TYPES,
    WHARDatasetID.MHEALTH: MHEALTH_SENSOR_TYPES,
    WHARDatasetID.DSADS: DSADS_SENSOR_TYPES,
    WHARDatasetID.KU_HAR: KU_HAR_SENSOR_TYPES,
    WHARDatasetID.HAR_SENSE: HAR_SENSE_SENSOR_TYPES,
}


def get_sensor_types(dataset_id: WHARDatasetID):
    if type(dataset_id) is str:
        dataset_id = WHARDatasetID(dataset_id)
    selected_dataset = datasets_to_types[dataset_id]

    sensor_locations = list(x[0].value for x in selected_dataset)
    sensor_types = list(x[1].value for x in selected_dataset)
    return sensor_locations, sensor_types



UCIHAR_IMU_GROUPS = [
    (SensorLocation.HIP, list(range(0, 9))),  # Hip IMU
]

PAMAP2_IMU_GROUPS = [
    (SensorLocation.ARMS, [0, 1, 2, 6, 7, 8, 9, 10, 11]),  # Hand IMU (mapped to ARMS)
    (SensorLocation.TORSO, [12, 13, 14, 18, 19, 20, 21, 22, 23]),  # Chest IMU
    (SensorLocation.LEGS, [24, 25, 26, 30, 31, 32, 33, 34, 35]),  # Ankle IMU
]

DSADS_IMU_GROUPS = [
    (SensorLocation.TORSO, list(range(0, 9))),  # Chest IMU
    (SensorLocation.ARMS, list(range(9, 18))),  # Right arm IMU
    (SensorLocation.ARMS, list(range(18, 27))),  # Left arm IMU
    (SensorLocation.LEGS, list(range(27, 36))),  # Right ankle IMU
    (SensorLocation.LEGS, list(range(36, 45))),  # Left ankle IMU
]

MHEALTH_IMU_GROUPS = [
    (SensorLocation.LEGS, list(range(5, 14))),
    (SensorLocation.ARMS, list(range(14, 23))),
]

# Opportunity (UCI) config selects channels in this order:
# IMU_BACK acc/gyro/mag (9), IMU_RUA (9), IMU_RLA (9), IMU_LUA (9), IMU_LLA (9),
# IMU_LSHOE (16), IMU_RSHOE (16).
# We map these to coarse body regions and always pick 9 channels per group.
OPPORTUNITY_IMU_GROUPS = [
    (SensorLocation.TORSO, list(range(0, 9))),  # IMU_BACK: acc/gyro/mag
    (SensorLocation.ARMS, list(range(9, 18))),  # IMU_RUA: acc/gyro/mag
    (SensorLocation.ARMS, list(range(18, 27))),  # IMU_RLA: acc/gyro/mag
    (SensorLocation.ARMS, list(range(27, 36))),  # IMU_LUA: acc/gyro/mag
    (SensorLocation.ARMS, list(range(36, 45))),  # IMU_LLA: acc/gyro/mag
    # Shoes: use eu(3) + nav_acc(3) + angvel_nav(3) = 9
    (SensorLocation.LEGS, [45, 46, 47, 48, 49, 50, 57, 58, 59]),  # IMU_LSHOE
    (SensorLocation.LEGS, [61, 62, 63, 64, 65, 66, 73, 74, 75]),  # IMU_RSHOE
]



datasets_to_imu_groups = {
    WHARDatasetID.PAMAP2: PAMAP2_IMU_GROUPS,
    WHARDatasetID.DSADS: DSADS_IMU_GROUPS,
    WHARDatasetID.UCI_HAR: UCIHAR_IMU_GROUPS,
    WHARDatasetID.MHEALTH: MHEALTH_IMU_GROUPS,
    WHARDatasetID.OPPORTUNITY: OPPORTUNITY_IMU_GROUPS,
}


def get_imu_groups(dataset_id: WHARDatasetID):
    if type(dataset_id) is str:
        dataset_id = WHARDatasetID(dataset_id)
    return datasets_to_imu_groups.get(dataset_id, {})
