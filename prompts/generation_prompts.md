

# MisstepMath Prompt List

Below is the standard prompt template used to generate new student mistakes across different grade levels, topics, sub-topics, and challenge types.

Each prompt includes:
- Grade
- Topic
- Sub-topic
- Challenge Type
- Example data from previous entries to ensure diversity

---

## Example Prompt

Generate 5–10 diverse student mistakes.

Grade: K5  
Topic: Data Analysis  
Sub-topic: Solve real-world word problems by referring to line plots  
Challenge Type: Misconception  

Using the following previous examples as a reference (for diversity, do not repeat them but extend the pattern):

```json
[
  {
    "challenge_type": "Misconception",
    "challenge_faced": "Confusing Mean and Mode",
    "example": "What is the mode of this data set?",
    "student_mistake": "Calculating the mean instead of identifying the mode.",
    "teachers_resolution_text_only": "Explain the difference between mean and mode with simple examples.",
    "teacher_response_whiteboard": {
      "whiteboard": "Label the data points clearly and circle the most frequent value.",
      "text": "The mode is the number that appears most often, not the average."
    }
  }
]
```

Prompt Instruction:
Create 5–10 new student mistake examples of the same challenge type that are on similar lines but not identical to the above.

Output format must be a valid JSON array with the following keys:
- challenge_type
- challenge_faced
- example
- student_mistake
- teachers_resolution_text_only
- teacher_response_whiteboard: { whiteboard, text }

---

This format was used consistently across all (grade, topic, sub-topic, challenge type) combinations in MisstepMath.
