import json
import os

from PIL import Image


def string_box(box):
    return str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3])


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def normalize_bbox(bbox, width, length):
    return [
        (1000 * bbox[0]) // width,
        (1000 * bbox[1]) // length,
        (1000 * bbox[2]) // width,
        (1000 * bbox[3]) // length,
    ]


def get_line_bbox(bboxs):
    x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
    y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

    x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

    assert x1 >= x0 and y1 >= y0
    bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
    return bbox


ocr_files = os.listdir("../../CORD/test/json")

print(ocr_files)


for file in ocr_files:
    ocr_file = os.path.join("../../CORD/test/json", file)
    result_file = os.path.join("../../CORD/test/gpt3_test", file)

    flag = 0
    temp_id = [0]
    with open(ocr_file, "r", encoding="utf-8") as f:
        with open(result_file, "w", encoding="utf-8") as fw:
            data = json.load(f)

            image_path = ocr_file.replace(".json", ".jpg").replace("json", "image")
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)

            width, length = image.size

            bboxes = []

            res_box_r = []
            res_label_r = []
            res_text_r = []
            res = ""
            res_rt = ""
            for item in data["valid_line"]:
                cur_line_bboxes = []
                line_words, label = item["words"], item["category"]
                line_words = [w for w in line_words if w["text"].strip() != ""]
                if len(line_words) == 0:
                    continue
                res_text = ""
                for idx, w in enumerate(line_words):
                    if idx >= 1:
                        res_text = res_text + " " + w["text"]
                    else:
                        res_text = w["text"]
                for w in line_words:
                    cur_line_bboxes.append(
                        normalize_bbox(quad_to_box(w["quad"]), width, length)
                    )

                cur_line_bboxes = get_line_bbox(cur_line_bboxes)

                text = res_text
                box = cur_line_bboxes[0]
                label = label.upper()

                box = string_box(box)
                # res += "{" + 'text:"{}",Box:[{}]'.format(text, box) + "}"
                res += json.dumps({"text": text, "Box": [box]})
                # For ground truth data, use the following line instead
                # res += ('{' + 'text:"{}",Box:[{}],entity:{}'.format(text, box, label) + '}')

            fw.write(res)
