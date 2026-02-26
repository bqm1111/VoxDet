learning_map = {
    "kitti": {
        0: 0,  # "unlabeled"
        1: 0,  # "outlier" mapped to "unlabeled" --------------------------mapped
        10: 1,  # "car"
        11: 2,  # "bicycle"
        13: 5,  # "bus" mapped to "other-vehicle" --------------------------mapped
        15: 3,  # "motorcycle"
        16: 5,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
        18: 4,  # "truck"
        20: 5,  # "other-vehicle"
        30: 6,  # "person"
        31: 7,  # "bicyclist"
        32: 8,  # "motorcyclist"
        40: 9,  # "road"
        44: 10,  # "parking"
        48: 11,  # "sidewalk"
        49: 12,  # "other-ground"
        50: 13,  # "building"
        51: 14,  # "fence"
        52: 0,  # "other-structure" mapped to "unlabeled" ------------------mapped
        60: 9,  # "lane-marking" to "road" ---------------------------------mapped
        70: 15,  # "vegetation"
        71: 16,  # "trunk"
        72: 17,  # "terrain"
        80: 18,  # "pole"
        81: 19,  # "traffic-sign"
        99: 0,  # "other-object" to "unlabeled" ----------------------------mapped
        252: 1,  # "moving-car" to "car" ------------------------------------mapped
        253: 7,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
        254: 6,  # "moving-person" to "person" ------------------------------mapped
        255: 8,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
        256: 5,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
        257: 5,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
        258: 4,  # "moving-truck" to "truck" --------------------------------mapped
        259: 5,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    },
    "tartanair": {
        0: 0,  # unlabeled -> unlabeled
        6: 1,  # cabinet -> cabinet
        7: 2,  # pipecover -> pipecover
        8: 3,  # metalpanel -> metalpanel
        9: 4,  # metalhandrail -> metalhandrail
        23: 5,  # light -> light
        27: 6,  # cieling -> cieling
        28: 7,  # metalplatform -> metalplatform
        36: 8,  # chair -> chair
        59: 9,  # cementcolumn -> cementcolumn
        60: 10,  # plug -> plug
        64: 11,  # ceiling -> ceiling
        65: 12,  # metalpole -> metalpole
        69: 13,  # ventpipe -> ventpipe
        70: 14,  # metalramp -> metalramp
        72: 15,  # car -> car
        116: 16,  # metalfloor -> metalfloor
        123: 17,  # toolbox -> toolbox
        132: 18,  # ventpipeclamp -> ventpipeclamp
        143: 19,  # metalstair -> metalstair
        146: 20,  # lamp -> lamp
        157: 21,  # tireassembly -> tireassembly
        160: 22,  # metalcieling -> metalcieling
        161: 23,  # platform -> platform
        171: 24,  # monitor -> monitor
        175: 25,  # wall -> wall
        180: 26,  # door -> door
        188: 27,  # sky -> sky
        191: 28,  # cable -> cable
        195: 29,  # chasis -> chasis
        199: 30,  # floor -> floor
        205: 31,  # table -> table
        208: 32,  # robotarm -> robotarm
        239: 33,  # keyboard -> keyboard
    },
}

# Class names for TartanAir (34 classes including unlabeled)
class_names = {
    "tartanair": [
        "unlabeled",  # 0
        "cabinet",  # 1
        "pipecover",  # 2
        "metalpanel",  # 3
        "metalhandrail",  # 4
        "light",  # 5
        "cieling",  # 6
        "metalplatform",  # 7
        "chair",  # 8
        "cementcolumn",  # 9
        "plug",  # 10
        "ceiling",  # 11
        "metalpole",  # 12
        "ventpipe",  # 13
        "metalramp",  # 14
        "car",  # 15
        "metalfloor",  # 16
        "toolbox",  # 17
        "ventpipeclamp",  # 18
        "metalstair",  # 19
        "lamp",  # 20
        "tireassembly",  # 21
        "metalcieling",  # 22
        "platform",  # 23
        "monitor",  # 24
        "wall",  # 25
        "door",  # 26
        "sky",  # 27
        "cable",  # 28
        "chasis",  # 29
        "floor",  # 30
        "table",  # 31
        "robotarm",  # 32
        "keyboard",  # 33
    ]
}

# Color map for visualization (BGR format)
color_map = {
    "tartanair": [
        [0, 0, 0],  # 0: unlabeled
        [153, 108, 6],  # 1: cabinet
        [112, 105, 191],  # 2: pipecover
        [89, 121, 72],  # 3: metalpanel
        [190, 225, 64],  # 4: metalhandrail
        [206, 190, 59],  # 5: light
        [81, 13, 36],  # 6: cieling
        [115, 176, 195],  # 7: metalplatform
        [161, 171, 27],  # 8: chair
        [135, 169, 180],  # 9: cementcolumn
        [29, 26, 199],  # 10: plug
        [102, 16, 239],  # 11: ceiling
        [242, 107, 146],  # 12: metalpole
        [156, 198, 23],  # 13: ventpipe
        [49, 89, 160],  # 14: metalramp
        [68, 218, 116],  # 15: car
        [11, 236, 9],  # 16: metalfloor
        [196, 30, 8],  # 17: toolbox
        [121, 67, 28],  # 18: ventpipeclamp
        [0, 53, 65],  # 19: metalstair
        [146, 52, 70],  # 20: lamp
        [226, 149, 143],  # 21: tireassembly
        [151, 126, 171],  # 22: metalcieling
        [194, 39, 7],  # 23: platform
        [205, 120, 161],  # 24: monitor
        [212, 51, 60],  # 25: wall
        [211, 80, 208],  # 26: door
        [189, 135, 188],  # 27: sky
        [54, 72, 205],  # 28: cable
        [103, 252, 157],  # 29: chasis
        [124, 21, 123],  # 30: floor
        [19, 132, 69],  # 31: table
        [195, 237, 132],  # 32: robotarm
        [94, 253, 175],  # 33: keyboard
    ]
}

# Number of classes
num_classes = {"tartanair": 34}
