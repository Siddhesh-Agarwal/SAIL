import json
import os

root_entity_test = "../../SROIE2019/test/entities"
root_ocr_test = "../../SROIE2019/test/box"

ori_res_ = "../../SROIE2019/test/result"
res_ = "../../SROIE2019/test/result_post"
# res_gt='../../SROIE2019/test/gpt3_test_cut_gt'

orifiles = os.listdir(ori_res_)

if not os.path.exists(res_):
    os.makedirs(res_)


# For a single file, convert the unformatted text generated by LLM into json format
def postpart(file):
    ori_res = os.path.join(ori_res_, file)
    res = os.path.join(res_, file.replace(".txt", ".json"))  # .replace('.txt','.json')
    js = {}
    with open(ori_res, "r", encoding="utf-8") as f:
        data = f.read().split("\n")
    for da in data:
        if "company" in da.lower():
            if "{" not in da:
                if "company" in da:
                    js["company"] = da.split("company:")[-1]
                else:
                    js["company"] = da.split("Company:")[-1]
                continue
            d = da.split("{")[-1].strip("}").strip(" ").strip('"')
            js["company"] = d.strip()
        if "address" in da.lower():
            if "{" not in da:
                if "address" in da:
                    js["address"] = da.split("address:")[-1]
                else:
                    js["address"] = da.split("Address:")[-1]
                continue
            ds = da.split("{")[1:]
            d = ""
            for k in ds:
                k = k.strip("}").strip(" ").strip('"')
                d += k + " "
            js["address"] = d.strip()
        if "date" in da.lower():
            if "{" not in da:
                if "date" in da:
                    js["date"] = da.split("date:")[-1]
                else:
                    js["date"] = da.split("Date:")[-1]
                continue
            ds = da.split("{")[1:]
            d = ""
            for k in ds:
                k = k.strip("}").strip(" ").strip('"')
                d += k + " "
            js["date"] = d.strip()
        if "total" in da.lower():
            if "{" not in da:
                if "total" in da:
                    js["total"] = da.split("total:")[-1]
                else:
                    js["total"] = da.split("Total:")[-1]
                continue
            ds = da.split("{")[1:]
            d = ""
            for k in ds:
                k = k.strip("}").strip(" ").strip('"')
                d += k + " "
            js["total"] = d.strip()
    with open(res, "w", encoding="utf-8") as f:
        json.dump(js, f)


def post():
    for file in orifiles:
        ori_res = os.path.join(ori_res_, file)
        res = os.path.join(res_, file)  # .replace('.txt','.json')
        with open(ori_res, "r", encoding="utf-8") as f:
            data = f.read()
        if data.count("{") != 1:
            print(file)
            postpart(file)
            continue
        data = data.split("{")[1].split("}")[0]
        data = "{" + data + "}"
        with open(res, "w", encoding="utf-8") as f:
            f.write(data)


post()
