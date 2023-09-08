from pathlib import Path

import fire
import openai

import dspy


def setup_dspy(key_path, colbert_url):
    with open(Path(key_path).expanduser()) as f:
        openai.api_key = f.read().strip()

    turbo = dspy.OpenAI(model="gpt-3.5-turbo")
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url=colbert_url)
    dspy.settings.configure(lm=turbo, rm=colbertv2_wiki17_abstracts)


class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def run_dspy_qa(
    question,
    key_path="~/.keys/openai_key.txt",
    colbert_url="http://20.102.90.50:2017/wiki17_abstracts",
):
    # Define the predictor.
    setup_dspy(key_path, colbert_url)
    generate_answer = dspy.Predict(BasicQA)

    example = "What is the nationality of the chef and restaurateur featured in Restaurant: Impossible?"
    # Call the predictor on a particular input.
    pred = generate_answer(question=example)
    print(pred)


if __name__ == "__main__":
    fire.Fire(run_dspy_qa)
