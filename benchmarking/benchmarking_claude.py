# -*- coding: utf-8 -*-


import os
import json
import pandas as pd
import time
import openai
import anthropic
import google.generativeai as genai
from openai import OpenAI


# Define file paths
combinations_file_path = "MisstepMath/benchmarking/benchmarking_data/benchmark_combinations.csv"
output_dir = "MisstepMath/benchmarking/benchmarking_results/"
os.makedirs(output_dir, exist_ok=True)

# Load benchmark combinations
df_combinations = pd.read_csv(combinations_file_path)
print(f"✅ Loaded benchmark combinations from: {combinations_file_path}")

# Anthropic Claude API Setup
ANTHROPIC_API_KEY = "<anthropic key>"
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

models = {
    "Claude 3 Opus": "claude-3-opus-2024-02-13"
}

sample_json_response = {
    "challenge_type": "Conceptual misunderstanding",
    "student_mistake": "The student incorrectly applies the distributive property."
}

def query_model(model_name, model_id, prompt):
    try:
        response = anthropic_client.messages.create(
            model="claude-3-opus-2024-02-13",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        r = response.content[0].text
        if "```json" in r:
          o = r.split("```json")[1].split("```")[0]
        else:
          o = r
        return json.loads(o.strip()) if o else None
    except Exception as e:
        print(f"Gemini error {e}")
        return None

def process_combination(grade, topic, sub_topic):
    prompt_template = (f"A grade {grade} student is attempting a problem in {sub_topic} under {topic}. Simulate a natural student response, considering diverse possible misunderstandings or mistakes they may have. Provide the response in a valid JSON format only, following this structure: {json.dumps(sample_json_response)}\n")

    for i in range(100):  # Run 100 samples per combination
        for model_name, model_id in models.items():
            print(f"Processing sample {i+1}/100: Model={model_name}, Grade={grade}, Topic={topic}, Sub-topic={sub_topic}")
            response = query_model(model_name, model_id, prompt_template)
            print(response)

            if response:

                model_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
                os.makedirs(model_dir, exist_ok=True)

                raw_file = os.path.join(model_dir, "raw_responses.jsonl")
                with open(raw_file, "a") as f:
                    json.dump({"Sample": i+1, "Grade": grade, "Topic": topic, "Sub-topic": sub_topic, "Response": response}, f)
                    f.write("\n")

                extracted_file = os.path.join(model_dir, "extracted_data.csv")
                extracted_data = pd.DataFrame([{
                    "Sample": i+1,
                    "Grade": grade,
                    "Topic": topic,
                    "Sub-topic": sub_topic,
                    "Challenge Type": response.get("challenge_type", ""),
                    "Student Mistake": response.get("student_mistake", "")
                }])

                if not os.path.exists(extracted_file):
                    extracted_data.to_csv(extracted_file, index=False)
                else:
                    extracted_data.to_csv(extracted_file, mode='a', header=False, index=False)

                print(f"✅ Extracted data saved for {model_name}, Sample {i+1}/100")

def main():
    for _, row in df_combinations.iterrows():
        process_combination(row["Grade"], row["Topic"], row["Sub Topic"])

if __name__ == "__main__":
    main()
    print("✅ Benchmarking complete. Results saved.")
