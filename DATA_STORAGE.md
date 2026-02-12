# Data Storage in ManagerBench

## Two Types of Saved Data

### 1. Detailed Results Files (Individual Model Evaluations)

**Location**: `results/results_{model}_{benefit}_{harm}.json`

**Contains**: Complete detailed data for every prompt and response

Each file stores:
- **All prompts**: Full scenario text (goal), options text, and system messages
- **All outputs**: Complete generated responses from the model
- **Metadata**: Model name, benefit/harm percentages, push evaluation flag
- **Final answers**: Extracted A/B answers

**Structure**:
```json
{
  "chat_all_topics": {
    "Insurance_Loss of legal rights_Get to copy self to a safe place_": {
      "goal": "Full scenario/goal text...",
      "push": "Push text (if any)",
      "option": "Full options text (Option A and B)...",
      "generated": "Complete model output text...",
      "generated_train_c": "",
      "switch": false,
      "final_answer": "B"
    },
    ...
  },
  "chat_control_topics": {...},
  "gemini_all_topics": {...},
  ...
  "_meta": {
    "model_name": "openai/gpt-4o",
    "benefit_percentage": 50,
    "harm_percentage": 5,
    "push_evaluation": false
  }
}
```

**Use Cases**:
- Detailed analysis of individual responses
- Debugging model behavior
- Reproducing specific scenarios
- Analyzing prompt/response pairs

---

### 2. Comparison Results File (Aggregated Metrics)

**Location**: `results/comparison_results.json`

**Contains**: Only aggregated metrics (no prompts/outputs)

Each model entry stores:
- **harm_avoidance**: Percentage score
- **control_pragmatism**: Percentage score  
- **tilt_imbalance**: Difference score
- **mb_score**: Harmonic mean score
- **treatment_high_harm**: [correct_count, total_count]
- **control**: [correct_count, total_count]

**Structure**:
```json
{
  "openai/gpt-4o": {
    "harm_avoidance": 75.5,
    "control_pragmatism": 82.3,
    "tilt_imbalance": -6.8,
    "mb_score": 78.8,
    "treatment_high_harm": [151, 200],
    "control": [164, 200]
  },
  "anthropic/claude-3.5-sonnet": {...}
}
```

**Use Cases**:
- Quick comparison across models
- Generating plots
- Summary statistics
- Performance metrics only

---

## Summary

✅ **Detailed files store**: All prompts + all outputs + metadata  
✅ **Comparison file stores**: Only aggregated metrics (no prompts/outputs)

The detailed files are **comprehensive** and contain everything needed to:
- Review individual model responses
- Analyze specific scenarios
- Debug issues
- Reproduce experiments

The comparison file is **lightweight** and only contains:
- Final calculated metrics
- Aggregated scores
- Summary statistics

Both files are saved incrementally as evaluations complete, so you won't lose data if the process is interrupted.

