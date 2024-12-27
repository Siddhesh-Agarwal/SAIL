import os
import heapq
from seqeval.metrics import precision_score, recall_score, f1_score

root_gt = "../../SROIE2019/test/gpt3_test_cut_gt"
root_pred = "../../SROIE2019/test/result_post"
res_path = "../../SROIE2019/test/result_eval.txt"

pred_file = os.listdir(root_pred)
gt_file = os.listdir(root_gt)


res_gt_list = [[] for _ in range(347)]  # len(pred_file)
res_preds_list = [[] for _ in range(347)]

gt_list = [[]]
preds_list = [[]]

restext = ""


class Node:
    def __init__(self, v):
        self.a, self.b = v[0], v[1]

    def __lt__(self, other):
        if self.b != other.b:
            return self.b < other.b


h = []
for i in range(347):
    resu_gt_file = gt_file[i]
    resu_pred_file = pred_file[i]
    with open(os.path.join(root_gt, resu_gt_file), "r", encoding="utf-8") as fgt:
        with open(
            os.path.join(root_pred, resu_pred_file), "r", encoding="utf-8"
        ) as fed:
            data_gt = fgt.read().replace("\n", "").split("}{")
            data_pred = fed.read()
            data_preds = data_pred.replace("\n", "").split("}{")

            for j in range(len(data_gt)):
                l1 = data_gt[j].split(":")[-1].strip("}")
                b1 = data_gt[j].split(",Box:")[1].split(",")[0]

                if b1 in data_pred:
                    l2 = data_pred.split(b1)[1].split("}")[0].split(":")[-1]
                else:
                    l2 = "other"
                if l1 != "other":
                    gt_list[0].append("B-" + l1)
                    res_gt_list[i].append("B-" + l1)
                else:
                    gt_list[0].append("O")
                    res_gt_list[i].append("O")
                if l2 != "other":
                    preds_list[0].append("B-" + l2)
                    res_preds_list[i].append("B-" + l2)
                else:
                    preds_list[0].append("O")
                    res_preds_list[i].append("O")

            results = {
                "precision": precision_score(gt_list, preds_list),
                "recall": recall_score(gt_list, preds_list),
                "f1": f1_score(gt_list, preds_list),
            }

            temp = []
            temp.append(i)
            temp.append(results["f1"])
            heapq.heappush(h, Node(temp))

            gt_list = [[]]
            preds_list = [[]]

res_results = {
    "precision": precision_score(res_gt_list, res_preds_list),
    "recall": recall_score(res_gt_list, res_preds_list),
    "f1": f1_score(res_gt_list, res_preds_list),
}
# print the F1 scores of whole dataset
print(res_results)
restext += str(res_results)

# sort
ans = heapq.nsmallest(347, h)

# print the F1 scores of files
for i in ans:
    idx = i.a
    resu_gt_file = gt_file[idx]
    print(gt_file[idx], i.b)
    restext += gt_file[idx] + " " + str(i.b) + "\n"

with open(res_path, "w", encoding="utf-8") as fl:
    fl.write(restext)
