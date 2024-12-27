import os


filepath = "../../CORD/train/gpt3_train_cut_gt"
files = os.listdir(filepath)

result_test_text = []
result_train_text = []


def check(num: str):
    flag = 0
    for char in num:
        if char.isalpha():
            flag = 1
    return flag


# Combine the words except numbers in the fragment to form document texts of the FUNSD dataset
def funsd_textprocess():
    testtext = ""
    print(len(files))
    for i in range(len(files)):
        file_test = files[i]
        ocr_file_test = os.path.join(filepath, file_test)
        with open(ocr_file_test, "r", encoding="utf-8") as f:
            datas = f.read().split("\n")
            for data in datas:
                res_test = ""
                entities = data.split("}{")
                for en in entities:
                    text = en.split("text:")[1].split(",Box")[0].strip('"')
                    if check(text) == 0:
                        continue
                    res_test = res_test + text + " "
                result_test_text.append(res_test)
                testtext = testtext + res_test + "\n"
            if len(datas) == 2:
                result_test_text.append("")
                testtext = testtext + "\n"
            if len(datas) == 1:
                result_test_text.append("")
                testtext = testtext + "\n"
                result_test_text.append("")
                testtext = testtext + "\n"

    with open("../processfiles/ptext_funsd_test.txt", "w", encoding="utf-8") as f:
        f.write(testtext)


# Combine the words except numbers in the fragment to form document texts of the CORD dataset
# The SROIE dataset use this function as well
def cord_textprocess():
    testtext = ""
    print(len(files))
    for i in range(len(files)):
        file_test = files[i]
        ocr_file_test = os.path.join(filepath, file_test)
        with open(ocr_file_test, "r", encoding="utf-8") as f:
            data = f.read()
            res_test = ""
            entities = data.split("}{")
            for en in entities:
                text = en.split("text:")[1].split(",Box")[0].strip('"')
                if check(text) == 0:
                    continue
                res_test = res_test + text + " "
            testtext = testtext + res_test

    with open("../processfiles/ptext_cord_train.txt", "w", encoding="utf-8") as f:
        f.write(testtext)


# Generate a file containing all entity information
def entitytext():
    testtext = ""
    print(len(files))
    for i in range(len(files)):
        file_test = files[i]
        ocr_file_test = os.path.join(filepath, file_test)
        with open(ocr_file_test, "r", encoding="utf-8") as f:
            datas = f.read().split("\n")
            for data in datas:
                if data == "":
                    continue
                entities = data.split("}{")
                for en in entities:
                    text = en.split("text:")[1].split(",Box")[0].strip('"')
                    if check(text) == 0:
                        continue
                    label = en.split("entity:")[1].strip("}")
                    box = en.split("Box:")[1].split(",entity")[0]
                    testtext = testtext + text + "|" + label + "|" + box + "\n"

    with open("../processfiles/pentitytext_cord_train.txt", "w", encoding="utf-8") as f:
        f.write(testtext)


# funsd_textprocess()
cord_textprocess()
entitytext()
