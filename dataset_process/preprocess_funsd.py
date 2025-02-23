import json
import os

import numpy as np
from XYcut import recursive_xy_cut


def string_box(box):
    # return str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3])
    return " ".join(map(str, box[:4]))


ocr_files = os.listdir("../../FUNSD/training_data/annotations")

print(ocr_files)

for file in ocr_files:
    ocr_file = os.path.join("../../FUNSD/training_data/annotations", file)
    result_filr = os.path.join("../../FUNSD/training_data/gpt_train_xycut", file)

    res1 = ""
    res2 = ""
    res3 = ""

    with open(ocr_file, "r", encoding="utf-8") as f:
        with open(result_filr, "w", encoding="utf-8") as fw:
            data = json.load(f)

            map_pixel = np.ones((1000, 1000), dtype=int) * (-1)

            idx = 0
            res_box = []
            res_label = []
            res_text = []
            res = []
            for i, item in enumerate(data["form"]):
                if item["text"] == "":
                    continue

                text = item["text"]
                boxes = item["box"]
                label = item["label"]

                x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
                res_box.append(boxes)
                res_label.append(label)
                res_text.append(text)

            random_boxes = np.array(res_box)
            random_text = np.array(res_text)
            random_label = np.array(res_label)
            recursive_xy_cut(
                np.asarray(random_boxes).astype(int), list(range(len(res_box))), res
            )

            sorted_boxes = random_boxes[np.array(res)]
            sorted_text = random_text[np.array(res)]
            sorted_label = random_label[np.array(res)]

            for text, boxes, label in zip(sorted_text, sorted_boxes, sorted_label):
                x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
                box = string_box(boxes)
                temp = json.dumps({"text": text, "Box": [box]})
                if (y1 + y2) / 2 <= 300:
                    res1 += temp
                elif (y1 + y2) / 2 >= 300 and (y1 + y2) / 2 <= 600:
                    res2 += temp
                else:
                    res3 += temp

                """
                # For ground truth data, use the following line instead
                if (y1 + y2) / 2 <= 300:
                    res1 += ('{' + 'text:"{}",Box:[{}],entity:{}'.format(text, box, label) + '}')
                elif (y1 + y2) / 2 >= 300 and (y1 + y2) / 2 <= 600:
                    res2 += ('{' + 'text:"{}",Box:[{}],entity:{}'.format(text, box, label) + '}')
                else:
                    res3 += ('{' + 'text:"{}",Box:[{}],entity:{}'.format(text, box, label) + '}')
                """

            res = res1 + "\n" + res2 + "\n" + res3
            fw.write(res)
