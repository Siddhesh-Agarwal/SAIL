import numpy as np
from PIL import Image
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# Calculate mean square error
def MSE(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2))) / 512


# Image Binarization
def binarynp(image, hash_size=512):
    image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    np_image = np.array(image)
    binary_image = (np_image < 122).astype(int)
    return np.array(binary_image)


# Find similar layout examples through layout images
def find_similar_images(test_file_path, train_path, num_images=4, reverse=True):
    image1 = Image.open(test_file_path)
    hash1 = binarynp(image1)
    hash1 = hash1.flatten()
    train_file = os.listdir(train_path)
    similarity_dict = {}
    for j in range(len(train_file)):
        image = Image.open(os.path.join(train_path, train_file[j]))
        hash2 = binarynp(image)
        hash2 = hash2.flatten()
        if 1 not in hash2:  # Omit blank images
            continue
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
    # Read the processed fragment text
    with open("../processfiles/ptext_funsd_test.txt", "r", encoding="utf-8") as f:
        result_test_text = f.read().split("\n")[:-1]
    with open("../processfiles/ptext_funsd_train.txt", "r", encoding="utf-8") as f:
        result_train_text = f.read().split("\n")[:-1]

    sentences = [result_test_text[k]]
    for i in range(len(result_train_text)):
        sentences.append(result_train_text[i])

    # encode the text
    embeddings = model.encode(sentences)

    # Choose the most similar fragment texts
    maxlist = []
    maxidx = []
    for i in range(1, len(sentences)):
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
    choosefile = sorted(
        choosefile, key=lambda x: x[1], reverse=reverse
    )  # True:descending
    choosefile = [x[0] for x in choosefile]
    print(choosefile)

    return choosefile


"""In order to reduce encoding time, the entity text in the training set is encoded only once, 
   and the encoding results are shared."""
with open("../processfiles/pentitytext_funsd_train2.txt", "r", encoding="utf-8") as f:
    result_train_text = f.read().split("\n")[:-1]
train_text = [a.split("|")[0] for a in result_train_text]
train_label = [a.split("|")[1] for a in result_train_text]
train_box = [a.split("|")[2] for a in result_train_text]
train_embeddings = model.encode(train_text)


# Find textually similar entities
def find_similar_entity(entities, result_num=4, reverse=True):
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
        choosefile = sorted(choosefile, key=lambda x: x[1], reverse=reverse)
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
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


# Read candidate label information
with open("funsdlabel.txt", "r", encoding="utf-8") as f:
    map_text = f.read()


# Generate prompt and input it into the LLM for inference
def predict(idx, testpn, num=4):
    pidx = idx * 3 + testpn
    test_file_path = (
        "../../FUNSD/testing_data/experience/layoutimage3/{}_{}.jpg".format(idx, testpn)
    )
    # Change to the address where the test layout images are stored
    train_path = "../../FUNSD/training_data/experience/layoutimage3"
    # Change to the address where the training layout images are stored
    # Find the layout examples
    similarimages = find_similar_images(test_file_path, train_path, num, True)
    # Find the textually similar document examples
    dtexample = find_similar_text(pidx, num, True)
    # Read the test data
    file = "{}.json".format(idx)
    with open(
        os.path.join("../../FUNSD/testing_data/gpt3_test_xycut_cut", file),
        "r",
        encoding="utf-8",
    ) as ft:
        data = ft.read().split("\n")

    if testpn >= len(data):  # The text corresponding to the given index is empty
        return "", ""

    # Layout demenstration
    ld = ""
    hd = ""
    for hf in similarimages:
        if hf[0].split("_")[1] == "0.jpg":
            pn = 0
        elif hf[0].split("_")[1] == "1.jpg":
            pn = 1
        else:
            pn = 2
        sifile = "{}.json".format(hf[0].split("_")[0])
        with open(
            os.path.join("../../FUNSD/training_data/gpt3_train_xycut_cut_gt", sifile),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read().split("\n")
            if pn >= len(h):
                continue
            h = h[pn]
            ld += "Document:" + h + "\n\n"
    # Ask the LLM to analyze the layout of the document
    task = "These are the information extracted from the document through OCR, and the Box is the position of the text in the document. Please analyze where each label is generally located in the document.\n"
    prompt_text = "Labels:\n" + map_text + "\n\n" + ld + task
    # complete layout demonstration
    la_text = ld + task + gpt_call(prompt_text)

    # Document-level text demonstrations
    for hf in dtexample:
        tf = "{}.json".format(int(int(hf) / 3))
        pn = int(hf) % 3
        with open(
            os.path.join("../../FUNSD/training_data/gpt3_train_xycut_cut", tf),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read().split("\n")
            if pn >= len(h):
                continue
            h = h[pn]
            if h == "":
                continue
            hd += "Q:" + h + ", What are the labels for these texts?\n"
        with open(
            os.path.join("../../FUNSD/training_data/gpt3_train_xycut_cut_gt", tf),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read().split("\n")[pn]
            hd += "A:" + h + "\n\n"

    # Entity-level text demonstrations
    data2 = data[testpn].split("}{")
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

    # generate prompt and input it into the LLM for inference
    prompt_text = ""
    temp_str = data[testpn] + ", What are the labels for these texts?"
    # print(temp_str)
    prompt_text = (
        "Labels:\n"
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
    res_text = gpt_call(prompt_text)

    # Simple format check. If the LLM output is not in the correct format, it will be re-issued.
    k = 0
    while "{" not in res_text:
        res_text = gpt_call(prompt_text)
        k += 1
        if k > 2:  # limit the number of re-issues
            break

    return res_text, similarentities


# Main Program
for idx in range(50):
    res = ""
    print(idx)
    for testpn in range(3):
        res_text, similarentities = predict(idx, testpn, num=4)
        print(res_text)
        res += res_text + "\n"
    with open(
        "../../FUNSD/testing_data/result/{}.txt".format(idx), "w", encoding="utf-8"
    ) as ft:
        ft.write(res)
