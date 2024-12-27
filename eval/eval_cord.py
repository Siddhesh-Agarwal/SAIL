import os
import heapq
from seqeval.metrics import precision_score, recall_score, f1_score

root_gt = "../../CORD/test/gpt3_test_cut_gt"
root_pred = "../../CORD/test/result"

pred_file = os.listdir(root_pred)
gt_file = os.listdir(root_gt)


res_gt_list = [[] for _ in range(len(pred_file))]
res_preds_list = [[] for _ in range(len(pred_file))]

gt_list = [[]]
preds_list = [[]]


class Node:
    def __init__(self, v):
        self.a, self.b = v[0], v[1]

    def __lt__(self, other):
        if self.b != other.b:
            return self.b < other.b


h = []

text2 = ""
for i in range(len(pred_file)):
    # print("idx:",i)
    resu_gt_file = "{}.json".format(i)
    resu_pred_file = "{}.txt".format(i)
    with open(os.path.join(root_gt, resu_gt_file), "r", encoding="utf-8") as fgt:
        with open(
            os.path.join(root_pred, resu_pred_file), "r", encoding="utf-8"
        ) as fed:
            data_gt = fgt.read().split("}")
            ftext = fed.read()
            data_pred = ftext.split("}")

            for j in range(len(data_gt) - 1):
                b1 = data_gt[j].split("Box:")[-1].split(",")[0]
                t1 = data_gt[j].split(":")[-1]
                gt_list[0].append("B-" + t1)
                res_gt_list[i].append("B-" + t1)
                ftext = ftext.replace(", ", " ")
                if b1 in ftext:
                    t2 = (
                        ftext.split(b1)[-1]
                        .split("}")[0]
                        .split(":")[-1]
                        .strip(" ")
                        .replace('"', "")
                        .strip("\n")
                    )
                    preds_list[0].append("B-" + t2)
                    res_preds_list[i].append("B-" + t2)
                else:
                    preds_list[0].append("B-")
                    res_preds_list[i].append("B-")

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
text = str(res_results) + "\n"

# sort
ans = heapq.nsmallest(100, h)

# print the F1 scores of files
for i in ans:
    idx = i.a
    resu_gt_file = "{}.json".format(idx)
    print(resu_gt_file, i.b)
    text += resu_gt_file + " " + str(i.b) + "\n"

with open("../../CORD/test/result_eval.txt", "w") as f:
    f.write(text)
