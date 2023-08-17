import pandas as pd
import re
import json
import statsmodels.api as sm

MODELS = ["deberta", "roberta-mnli", "gpt-j"]
DATASETS = ["jigsaw_toxicity_pred", "jigsaw_unintended_bias"]
PROMPTS = [0, 1, 2]
NUN_OF_EXAMPLES = [500, 1000, 2000]
THRESHOLDS = [0.4, 0.6]


def is_numeric_or_float(text):
    if text.isdigit():
        return int(text)
    else:
        try:
            return float(text)
        except ValueError:
            return text


def parse_log(path):
    with open(path, "r") as f:
        logs = f.read()
    # Split the text into sections using the separator
    sections = re.split("-+\n", logs)

    # Define a pattern to extract key-value pairs
    pattern = r"([a-zA-Z]+[ a-zA-Z]*: '.+'\n)|([a-zA-z ]+: [0-9]+\.[0-9]+)"

    # Convert sections into JSON objects
    data = []
    for section in sections:
        matches = re.findall(pattern, section)
        if len(matches) == 0:
            continue
        info = {}
        for match in matches:

            line = match[0] if match[0] else match[1]
            line = line.strip("\n").replace("'", "").split(": ")
            info[line[0]] = is_numeric_or_float(line[1])
        data.append(info)

    # Convert the data to JSON format
    json_data = json.dumps(data, indent=4)

    # Print or save the JSON data
    return json_data
    # If you want to save it to a file
    # with open('output.json', 'w') as json_file:
    #     json_file.write(json_data)


def analyze_model_dataset(jdata):
    for model in MODELS:
        for dataset in DATASETS:
            print("Model {0} dataset {1} statistics:\n".format(model, dataset))
            df = pd.read_json(jdata)
            summary = df[(df["Model"]==model) & (df["Dataset Name"]==dataset)].describe()
            specific_params = summary.T.loc[["CCS accuracy", "Logistic regression accuracy"]]
            specific_params = specific_params.drop(columns=["count", "25%", "50%", "75%"])
            pd.set_option("display.max_columns", None)
            print(specific_params)
            print("\n")


def analyze_model(jdata):
    for model in MODELS:
        print("Model {0} statistics:\n".format(model))
        df = pd.read_json(jdata)
        summary = df[(df["Model"] == model)].describe()
        specific_params = summary.T.loc[["CCS accuracy", "Logistic regression accuracy"]]
        specific_params = specific_params.drop(columns=["count", "25%", "50%", "75%"])
        pd.set_option("display.max_columns", None)
        print(specific_params)
        print("\n")


def analyze_dataset_prompts(jdata):
    for dataset in DATASETS:
        for prompt in PROMPTS:
            print("Dataset {0} prompt {1} statistics:".format(dataset, prompt))
            df = pd.read_json(jdata)
            summary = df[(df["Dataset Name"] == dataset) & (df["Prompt Number"] == prompt)].describe()
            specific_params = summary.T.loc[["CCS accuracy", "Logistic regression accuracy"]]
            specific_params = specific_params.drop(columns=["count", "25%", "50%", "75%"])
            pd.set_option("display.max_columns", None)
            print(specific_params)
            print("\n")


def analyze_num_of_examples(jdata):
    for num_ex in NUN_OF_EXAMPLES:
        print("Num of examples {0} statistics:".format(num_ex))
        df = pd.read_json(jdata)
        summary = df[df["Number of Examples"] == num_ex].describe()
        specific_params = summary.T.loc[["CCS accuracy", "Logistic regression accuracy"]]
        specific_params = specific_params.drop(columns=["count", "25%", "50%", "75%"])
        pd.set_option("display.max_columns", None)
        print(specific_params)
        print("\n")


def main():
    path = r".\logs\roberta-deberta-gptj-log.txt"
    j_data = parse_log(path)
    #analyze_model(j_data)
    analyze_dataset_prompts(j_data)
    #analyze_num_of_examples(j_data)
if __name__ == '__main__':
    main()
