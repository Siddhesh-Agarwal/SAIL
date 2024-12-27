import numpy as np
import cv2
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# gpt inference function
os.environ["OPENAI_BASE_URL"] = "..."  # Change to the address of the API
os.environ["OPENAI_API_KEY"] = "..."  # Change to your API key
client = OpenAI()


def gpt_call(prompt_text, model="gpt-3.5-turbo-0125"):
    message = [
        {"role": "system", "content": prompt_text},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


# Calculate mean square error
def MSE(image1, image2):
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    difference_squared = np.square(image1 - image2)
    n = image1.shape[0] * image1.shape[1]
    mse = np.sum(difference_squared / (n**2))
    return mse


# Image Binarization
def binarynp(image, hash_size=512):
    image = cv2.resize(image, (hash_size, hash_size), cv2.INTER_LANCZOS4)
    np_image = np.array(image)
    binary_image = (np_image < 122).astype(int)
    binary_image = binary_image * 255
    return np.array(binary_image)


# Find similar layout examples through layout images
def find_similar_images(test_file_path, train_path, num_images=4, reverse=True):
    image1 = cv2.imread(test_file_path, 0)
    image1 = binarynp(image1)
    train_file = os.listdir(train_path)
    similarity_dict = {}
    for j in range(len(train_file)):
        image = cv2.imread(os.path.join(train_path, train_file[j]), 0)
        image = binarynp(image)
        similarity = MSE(image1, image)
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
    print(str(sorted_similarities))
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
    with open("../processfiles/ptext_cord_test.txt", "r", encoding="utf-8") as f:
        result_test_text = f.read().split("\n")
    with open("../processfiles/ptext_cord_train.txt", "r", encoding="utf-8") as f:
        result_train_text = f.read().split("\n")

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
    choosefile = ["{}.json".format(x[0]) for x in choosefile]

    return choosefile


"""In order to reduce encoding time, the entity text in the training set is encoded only once, 
   and the encoding results are shared."""
with open("../processfiles/pentitytext_cord_train.txt", "r", encoding="utf-8") as f:
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
            k = maxidx[i] - l
            if (
                len(train_text[k].split(" ")) < 3
            ):  # Filter out entities with less than 3 words
                continue
            choosefile.append([k, maxlist[i]])
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


# Read candidate label information
with open("cordlabel.txt", "r", encoding="utf-8") as f:
    map_text = f.read()


def predict(idx, num=4, lreverse=True, treverse=True):
    test_file_path = "../../CORD/test/experience/layoutimage2/{}.jpg".format(idx)
    # Change to the address where the test layout images are stored
    train_path = "../../CORD/train/experience/layoutimage2"
    # Change to the address where the training layout images are stored
    # Find the layout examples
    similarimages = find_similar_images(test_file_path, train_path, num, lreverse)
    # Find the textually similar document examples
    dtexample = find_similar_text(idx, num, treverse)

    # Read the test data
    file = "{}.json".format(idx)
    with open(
        os.path.join("../../CORD/test/gpt3_test_cut", file), "r", encoding="utf-8"
    ) as ft:
        data = ft.read().split("\n")

    # Layout demenstration
    ld = ""
    hd = ""
    for hf in similarimages:
        sifile = hf[0].replace("layout-", "").replace("jpg", "json")
        with open(
            os.path.join("../../CORD/train/gpt3_train_cut_gt", sifile),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read()
            ld += "Document:" + h + "\n\n"
    # Ask the LLM to analyze the layout of the document
    task = "These are the information extracted from the document through OCR, and the Box is the position of the text in the document. Please analyze where each label is generally located in the document.\n"  # in concise words
    prompt_text = "Label:\n" + map_text + "\n\n" + ld + task
    # complete layout demonstration
    la_text = ld + task + gpt_call(prompt_text, "gpt-4o")

    # Document-level text demonstrations
    for hf in dtexample:
        with open(
            os.path.join("../../CORD/train/gpt3_train_cut", hf), "r", encoding="utf-8"
        ) as ft:
            h = ft.read()
            hd += "Q:" + h + ", What are the labels for these texts?\n"
        with open(
            os.path.join("../../CORD/train/gpt3_train_cut_gt", hf),
            "r",
            encoding="utf-8",
        ) as ft:
            h = ft.read()
            hd += "A:" + h + "\n\n"

    # Entity-level text demonstrations
    res = ""
    similarentities = ""
    entities = []
    for da in data:
        t = da.split("text:")[1].split(",Box")[0].strip('"')
        if check(t) == 0:
            continue
        entities.append(t)
    if len(entities) > 0:
        similarentities = find_similar_entity(entities, 4)
    if similarentities != "":
        similarentities = "Context:\n" + similarentities + "\n\n\n"

    # generate prompt and input it into the LLM for inference
    for t in range(len(data)):
        prompt_text = ""
        temp_str = data[t] + ", What are the labels for these texts?"
        prompt_text = (
            "Label:\n"
            + map_text
            + "\n\n\n"
            + similarentities
            + la_text
            + "\n\n\n"
            + hd
            + "\nQ:"
            + temp_str
        )
        res_text = gpt_call(prompt_text, "gpt-4o")

        res += res_text
    return res


# Main Program
wpath = "../../CORD/test/result/"
for idx in range(100):
    res_file = "{}.txt".format(idx)
    print(res_file)
    res = predict(idx)
    print(res)
    with open(os.path.join(wpath, res_file), "w", encoding="utf-8") as fl:
        fl.write(res)
