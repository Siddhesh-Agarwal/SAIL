import numpy as np
from PIL import Image
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tiktoken
import random

# To calculate the number of tokens
enc = tiktoken.get_encoding("cl100k_base")

# gpt inference function
os.environ["OPENAI_BASE_URL"] = "..."  # Change to the address of the API
os.environ["OPENAI_API_KEY"] = "..."  # Change to your API key
client = OpenAI()


def gpt_call(prompt_text):
    message = [
        {"role": "system", "content": prompt_text},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=message,
        temperature=0,
        max_tokens=1100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


# Calculate mean square error
def MSE(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2))) / 512


# Image Binarization
def binarynp(image, hash_size=512):
    image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    np_image = np.array(image)
    binary_image = (np_image < 122).astype(int)
    return np.array(binary_image)


# Generate ground truth in prompt
def generategt(text):
    entities = text.split("}{")
    companytext = '"company":'
    addresstext = '"address":'
    datetext = '"date":'
    totaltext = '"total":'
    for en in entities:
        label = en.split("entity:")[-1]
        entext = en.split("text:")[1].split(",Box")[0]
        if label == "company":
            companytext = companytext + "{" + entext + "}"
        if label == "address":
            addresstext = addresstext + "{" + entext + "}"
        if label == "date":
            datetext = datetext + "{" + entext + "}"
        if label == "total":
            if entext not in totaltext:
                totaltext = totaltext + "{" + entext + "}"
    res = companytext + "\n" + addresstext + "\n" + datetext + "\n" + totaltext + "\n"
    return res


# Find similar layout examples through layout images
def find_similar_images(test_file_path, train_path, num_images=4, reverse=False):
    image1 = Image.open(test_file_path)
    hash1 = binarynp(image1)
    hash1 = hash1.flatten().T
    train_file = os.listdir(train_path)
    similarity_dict = {}
    for j in range(len(train_file)):
        image = Image.open(os.path.join(train_path, train_file[j]))
        hash2 = binarynp(image)
        hash2 = hash2.flatten().T
        similarity = MSE(hash1, hash2)
        similarity_dict[train_file[j]] = similarity

    # Choose the most similar images
    sorted_similarities = sorted(
        similarity_dict.items(), key=lambda x: x[1], reverse=False
    )
    sorted_similarities = sorted_similarities[:num_images]
    # Order in the prompt
    sorted_similarities = sorted(
        sorted_similarities, key=lambda x: x[1], reverse=reverse
    )
    print(sorted_similarities)
    return sorted_similarities[:num_images]


model = SentenceTransformer("all-MiniLM-L6-v2")


def check(num):
    flag = 0
    for i in range(len(num)):
        if num[i] >= "a" and num[i] <= "z" or num[i] >= "A" and num[i] <= "Z":
            flag = 1
    return flag


def find_similar_text(k, result_num=4, reverse=True):
    # Read the processed document text
    with open("../processfiles/ptext_sroie_test.txt", "r", encoding="utf-8") as f:
        result_test_text = f.read().split("\n")[:-1]
    with open("../processfiles/ptext_sroie_train.txt", "r", encoding="utf-8") as f:
        result_train_text = f.read().split("\n")[:-1]

    sentences = [result_test_text[k]]
    for i in range(len(result_train_text)):
        sentences.append(result_train_text[i])

    # encode the text
    embeddings = model.encode(sentences)

    # Choose the most similar fragment texts
    maxlist = []
    maxidx = []
    for i in range(1, len(result_train_text) + 1):
        cos_sim = cosine_similarity(
            embeddings[0].reshape(1, -1), embeddings[i].reshape(1, -1)
        )
        if len(maxlist) < result_num:
            maxlist.append(cos_sim[0][0])
            maxidx.append(i)
        elif cos_sim[0][0] > min(maxlist):
            maxlist[maxlist.index(min(maxlist))] = cos_sim[0][0]
            maxidx[maxlist.index(min(maxlist))] = i
    choosefile = []
    for i in range(len(maxlist)):
        choosefile.append([maxidx[i] - 1, maxlist[i]])
    # Order in the prompt
    choosefile = sorted(choosefile, key=lambda x: x[1], reverse=reverse)  # True:降序
    choosefile = [x[0] for x in choosefile]

    return choosefile


"""In order to reduce encoding time, the entity text in the training set is encoded only once, 
   and the encoding results are shared."""
with open("../processfiles/pentitytext_sroie_train2.txt", "r", encoding="utf-8") as f:
    result_train_text = f.read().split("\n")[:-1]
train_text = [a.split("|")[0] for a in result_train_text]
train_label = [a.split("|")[1] for a in result_train_text]
train_box = [a.split("|")[2] for a in result_train_text]
train_embeddings = model.encode(train_text)


# Find textually similar entities
def find_similar_entity(entities, result_num=4, reverse=True):  # 加入box信息
    restext = ""
    # encode the entity texts
    sentences = entities
    embeddings = model.encode(sentences)
    embeddings = np.concatenate((embeddings, train_embeddings), axis=0)

    # Choose the most similar entity texts
    l = len(entities)
    cos_sims = cosine_similarity(
        embeddings[:l], embeddings[l : len(result_train_text) + l]
    )
    for k in range(l):
        maxlist = []
        maxidx = []
        for i in range(l, len(result_train_text) + l):
            cos_sim = cos_sims[k][i - l]
            if len(maxlist) < result_num:
                maxlist.append(cos_sim)
                maxidx.append(i)
            elif cos_sim > min(maxlist):
                maxlist[maxlist.index(min(maxlist))] = cos_sim
                maxidx[maxlist.index(min(maxlist))] = i
        choosefile = []
        for i in range(len(maxlist)):
            choosefile.append([maxidx[i] - l, maxlist[i]])
        # Order in the prompt
        choosefile = sorted(
            choosefile, key=lambda x: x[1], reverse=reverse
        )  # True:降序
        # write the prompt
        for i in range(len(choosefile)):
            restext += (
                '{text:"'
                + train_text[choosefile[i][0]]
                + '",Box:'
                + train_box[choosefile[i][0]]
                + ",entity:"
                + train_label[choosefile[i][0]]
                + "}\n"
            )

    return restext


# Candidate label information
map_text = '"company","address","date","total"'


def predict(idx, num=4, lreverse=True, treverse=True):
    test_file_path = "../../SROIE2019/test/experience/layoutimage"
    # Change to the address where the test layout images are stored
    test_files = os.listdir(test_file_path)
    test_file = test_file_path + "/" + test_files[idx]
    train_path = "../../SROIE2019/train/experience/layoutimage"
    # Change to the address where the training layout images are stored
    train_files = os.listdir("../../SROIE2019/train/gpt3_train_cut_gt")
    # Find the layout examples
    similarimages = find_similar_images(test_file, train_path, num, lreverse)
    # Find the textually similar document examples
    dtexample = find_similar_text(idx, num, treverse)

    # Read the test data
    file = test_files[idx].replace("jpg", "txt")
    with open(
        os.path.join("../../SROIE2019/test/gpt3_test_cut", file), "r", encoding="utf-8"
    ) as ft:
        data = ft.read()

    # Layout demenstration
    ld = ""
    hd = ""
    for hf in similarimages:
        sifile = hf[0].replace("jpg", "txt")
        with open(
            os.path.join("../../SROIE2019/train/gpt3_train_cut_gt", sifile),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read()
            ld += "Document:" + h + "\n\n"
    # Ask the LLM to analyze the layout of the document
    task = "These are the information extracted from the document through OCR, and the Box is the position of the text in the document. Please analyze where each label is generally located in the document.\n"
    prompt_text = "Label:\n" + map_text + "\n\n" + ld + task
    # complete layout demonstration
    la_text = ld + task + gpt_call(prompt_text)

    # Document-level text demonstrations
    for hf in dtexample:
        f = train_files[int(hf)]
        with open(
            os.path.join("../../SROIE2019/train/gpt3_train_cut", f),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read()
            hd += (
                "Q:"
                + h
                + ", return one company and its original address, one total, and one date?\n"
            )
        with open(
            os.path.join("../../SROIE2019/train/gpt3_train_cut_gt", f),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read()
            h = generategt(h)
            hd += "A:" + h + "\n\n"

    # Entity-level text demonstrations
    data2 = data.split("}{")
    similarentities = ""
    entities = []
    for da in data2:
        t = da.split("text:")[1].split(",Box")[0].strip('"')
        if check(t) == 0:
            continue
        entities.append(t)
    if len(entities) > 0:
        similarentities = find_similar_entity(entities, 4)
    if similarentities != "":
        similarentities = "Context:\n" + similarentities

    # generate prompt
    res = ""
    prompt_text = ""
    temp_str = (
        data + ", return one company and its original address, one total, and one date?"
    )
    prompt_text = (
        "Label:\n"
        + map_text
        + "\n\n\n"
        + similarentities
        + "\n\n\n"
        + la_text
        + "\n\n\n"
        + hd
        + "\nQ:"
        + temp_str
    )
    prompt_text2 = (
        "Label:\n" + map_text + "\n\n\n" + la_text + "\n\n\n" + hd + "\nQ:" + temp_str
    )

    # Dealing with the situation where the prompt word is too long
    if len(enc.encode(prompt_text2)) > 15000:
        print("2")
        return predict(idx, 2)
    while len(enc.encode(prompt_text)) > 15000:
        print(idx)
        sen = similarentities.split("\n")
        idxs = sorted(random.sample(range(1, len(sen)), int(len(sen) * 0.7)))
        similarentities = "Context:\n"
        for i in idxs:
            similarentities += sen[i] + "\n"
        prompt_text = (
            "Label:\n"
            + map_text
            + "\n\n\n"
            + similarentities
            + "\n\n\n"
            + la_text
            + "\n\n\n"
            + hd
            + "\nQ:"
            + temp_str
        )
    # Input it into the LLM for inference
    res_text = gpt_call(prompt_text)
    res += res_text
    return res


# Main Program
wpath = "../../SROIE2019/test/result/"
test_file_path = "../../SROIE2019/test/experience/layoutimage"
test_files = os.listdir(test_file_path)
print(len(test_files))
for idx in range(len(test_files)):
    res_file = test_files[idx].replace(".jpg", ".txt")
    print(res_file)
    print(idx)
    res1 = predict(idx, 4)
    with open(os.path.join(wpath, res_file), "w", encoding="utf-8") as fl:
        fl.write(res1)
