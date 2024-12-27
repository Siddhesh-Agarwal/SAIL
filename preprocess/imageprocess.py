import os
from PIL import Image, ImageDraw


def layout_image(input_root, output_root):
    files = os.listdir(input_root)
    for i in range(len(files)):
        print(i)
        with open(os.path.join(input_root, files[i]), "r", encoding="utf-8") as ft:
            datas = ft.read().replace("\n", "").split("\n")
            x_min = 1000
            x_max = 0
            y_mins = []
            y_maxs = []
            for k in range(len(datas)):
                data = datas[k]
                if data == "":  # Blank Image
                    y_mins.append(0)
                    y_maxs.append(0)
                    image = Image.new("RGB", (100, 100), 255)
                    outputfile = (
                        files[i].replace(".json", "") + ".jpg"
                    )  # +'_'+str(k)+'.jpg' funsd
                    image.save(os.path.join(output_root, outputfile))
                    continue
                data = data.split("}")[:-1]
                y_min = 1000
                y_max = 0
                # Finding the boundaries of informative areas
                for j in range(len(data)):
                    b1 = data[j].split("Box:")[-1].split(",")[0]
                    b1 = b1.replace("[", "").replace("]", "").split(" ")
                    b1 = [int(b) for b in b1]  # x1 y1 x2 y2
                    if b1[0] < x_min:
                        x_min = b1[0]
                    if b1[2] > x_max:
                        x_max = b1[2]
                    if b1[1] < y_min:
                        y_min = b1[1]
                    if b1[3] > y_max:
                        y_max = b1[3]
                print(x_max - x_min, y_min)
                y_mins.append(y_min)
                y_maxs.append(y_max)
            # generate layout image
            for k in range(len(datas)):
                data = datas[k]
                if data == "":
                    continue
                data = data.split("}")[:-1]
                image = Image.new(
                    "RGB", (x_max - x_min + 20, y_maxs[k] - y_mins[k] + 20), 255
                )
                # Create a drawing object
                draw = ImageDraw.Draw(image)
                color = (0, 0, 0)  # black

                for j in range(len(data)):
                    b1 = data[j].split("Box:")[-1].split(",")[0]
                    b1 = b1.replace("[", "").replace("]", "").split(" ")
                    # print(b1)
                    b1 = [int(b) for b in b1]
                    b1[0] = b1[0] - x_min + 10
                    b1[1] = b1[1] - y_mins[k] + 10
                    b1[2] = b1[2] - x_min + 10
                    b1[3] = b1[3] - y_mins[k] + 10
                    # print(b1)
                    # paint black rectangle
                    draw.rectangle(b1, fill=color)

                outputfile = (
                    files[i].replace(".json", "") + ".jpg"
                )  # +'_'+str(k)+'.jpg' funsd
                image.save(os.path.join(output_root, outputfile))


input_root = "../../CORD/train/gpt3_train_cut_gt"
output_root = "../../CORD/train/layoutimage"
layout_image(input_root, output_root)
