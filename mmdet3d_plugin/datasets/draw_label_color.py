import cv2
import numpy as np

CLASS_NAMES = {
    "cabinet": 6,
    "cable": 191,
    "car": 72,
    "ceiling": 64,
    "cementcolumn": 59,
    "chair": 36,
    "chasis": 195,
    "cieling": 27,
    "door": 180,
    "floor": 199,
    "keyboard": 239,
    "lamp": 146,
    "light": 23,
    "metalcieling": 160,
    "metalfloor": 116,
    "metalhandrail": 9,
    "metalpanel": 8,
    "metalplatform": 28,
    "metalpole": 65,
    "metalramp": 70,
    "metalstair": 143,
    "monitor": 171,
    "pipecover": 7,
    "platform": 161,
    "plug": 60,
    "robotarm": 208,
    "sky": 188,
    "table": 205,
    "tireassembly": 157,
    "toolbox": 123,
    "ventpipe": 69,
    "ventpipeclamp": 132,
    "wall": 175,
}
LEARNING_MAP = {
    0: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    23: 5,
    27: 6,
    28: 7,
    36: 8,
    59: 9,
    60: 10,
    64: 11,
    65: 12,
    69: 13,
    70: 14,
    72: 15,
    116: 16,
    123: 17,
    132: 18,
    143: 19,
    146: 20,
    157: 21,
    160: 22,
    161: 23,
    171: 24,
    175: 25,
    180: 26,
    188: 27,
    191: 28,
    195: 29,
    199: 30,
    205: 31,
    208: 32,
    239: 33,
}

COLOR_MAP = [
    [0, 0, 0],
    [153, 108, 6],
    [112, 105, 191],
    [89, 121, 72],
    [190, 225, 64],
    [206, 190, 59],
    [81, 13, 36],
    [115, 176, 195],
    [161, 171, 27],
    [135, 169, 180],
    [29, 26, 199],
    [102, 16, 239],
    [242, 107, 146],
    [156, 198, 23],
    [49, 89, 160],
    [68, 218, 116],
    [11, 236, 9],
    [196, 30, 8],
    [121, 67, 28],
    [0, 53, 65],
    [146, 52, 70],
    [226, 149, 143],
    [151, 126, 171],
    [194, 39, 7],
    [205, 120, 161],
    [212, 51, 60],
    [211, 80, 208],
    [189, 135, 188],
    [54, 72, 205],
    [103, 252, 157],
    [124, 21, 123],
    [19, 132, 69],
    [195, 237, 132],
    [94, 253, 175],
]

color_pair = {}
for k, v in CLASS_NAMES.items():
    color_pair[k] = COLOR_MAP[LEARNING_MAP[v]]

color_pair["unlabel"] = [0, 0, 0]
img = np.zeros((480, 1280, 3))
img[:] = [128, 128, 160]
h, w, c = img.shape
distance_x = w // 9
num_class = len(color_pair.keys())
start_x = (w - distance_x * 7) // 2
start_y = 100
distance_y = 80
box_w = 10
box_h = 10
for i, (name, color) in enumerate(color_pair.items()):
    y = i // 8
    x = i % 8
    center_x = start_x + x * distance_x
    center_y = start_y + y * distance_y
    text_length = len(name)
    cv2.putText(
        img,
        name,
        (center_x - text_length // 2 * 8, center_y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
        False,
    )
    cv2.rectangle(
        img,
        (center_x - box_w, center_y - box_h),
        (center_x + box_w, center_y + box_h),
        color,
        -1,
    )
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.uint8)

cv2.imshow("img", img)
cv2.waitKey()
