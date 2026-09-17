# cell 1
# Rubric mapping: Code Submission (11)
# 1. Install NumPy FIRST and force the older version to prevent binary incompatibility errors
!pip install -q "numpy<2.0.0" --force-reinstall

# 2. Install all other dependencies with specific versions for reproducibility
!pip install -q transformers==4.46.0 datasets==2.21.0 accelerate==1.0.1 evaluate==0.4.3
!pip install -q sacrebleu==2.4.3 bert-score==0.3.13 peft==0.13.2 bitsandbytes==0.44.1
!pip install -q sentencepiece==0.2.0 pandas==2.2.3 matplotlib==3.9.2 scikit-learn==1.5.2

import torch
import sys
import platform
import numpy as np

print("-" * 50)
print(f"Python Version: {sys.version}")
print(f"NumPy Version: {np.__version__}") # Verify this is 1.26.x
print(f"PyTorch Version: {torch.__version__}")

# GPU Check & Assertion
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device Capability: {torch.cuda.get_device_capability(0)}")
    # Assert T4 or better for LoRA/bitsandbytes compatibility
    assert torch.cuda.get_device_capability(0)[0] >= 3, "GPU capability too low for efficient PEFT training."
else:
    print("WARNING: No GPU detected.")
    print("ACTION REQUIRED: Go to Runtime -> Change runtime type -> T4 GPU")

print("-" * 50)
print("Environment setup complete.")

# cell 2
# Rubric mapping: Code Submission (11), Fine-Tune strategy (7)
import os
import random
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CFG:
    # Model Architecture
    model_name: str = "google/flan-t5-small" # Efficient for T4, switch to 'base' if resources allow

    # Data / Tokenizer
    max_source_len: int = 512
    max_target_len: int = 128

    # Training Hyperparameters (T4 Friendly)
    learning_rate: float = 3e-4 # Standard for fine-tuning
    batch_size: int = 8
    epochs: int = 3 # PDF requires explicit epochs
    grad_accum_steps: int = 4 # Effective batch size = 32
    warmup_steps: int = 100
    seed: int = 42

    # LoRA Config (PEFT)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Paths
    output_dir: str = "/content/outputs"
    metrics_dir: str = "/content/metrics"
    model_save_dir: str = "/content/models"

def set_seed(seed=42):
    """Sets the seed for reproducibility across torch, numpy, and python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")

# Initialize
config = CFG()
set_seed(config.seed)

# Create directory structure
for d in [config.output_dir, config.metrics_dir, config.model_save_dir]:
    os.makedirs(d, exist_ok=True)
    print(f"Created/Verified directory: {d}")

# cell 3
# Rubric mapping: Fine-Tune strategy (7), Coursework goal (1)
from datasets import load_dataset, DatasetDict

def load_amazon_summarization_data():
    """
    Loads Amazon Polarity and maps it for a summarization task.
    Input: 'content' (the review)
    Target: 'title' (the review summary/headline)
    """
    print("Loading Amazon Polarity dataset from Hugging Face Hub...")

    # We load a small subset (10,000) to keep T4 memory safe and training fast
    # 'train[:10000]' ensures reproducibility and prevents OOM errors
    raw_data = load_dataset("amazon_polarity", split="train[:10000]")

    # 1. Standardize column names to 'text' and 'summary' to match the pipeline
    # content -> text | title -> summary
    dataset = raw_data.map(lambda x: {
        "text": x["content"],
        "summary": x["title"]
    }, remove_columns=["content", "title", "label"])

    # 2. Train/Validation/Test Split (80/10/10)
    # Using fixed seed from our CFG class
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=config.seed)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=config.seed)

    final_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

    print(f"Dataset Split: Train({len(final_dataset['train'])}), Val({len(final_dataset['validation'])}), Test({len(final_dataset['test'])})")
    return final_dataset

# Execute the loader
raw_dataset = load_amazon_summarization_data()

# Verification print
print("-" * 30)
print("Example Review (Input):", raw_dataset['train'][0]['text'][:150] + "...")
print("Example Title (Target):", raw_dataset['train'][0]['summary'])

#cell 4
# Rubric mapping: Baseline setup (3), Prompt Engineering (5)
from transformers import AutoTokenizer

print(f"Loading tokenizer for: {config.model_name}")
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

def preprocess_function(examples):
    """
    Standard seq2seq preprocessing for Amazon Reviews:
    Input: 'text' (Review Content)
    Target: 'summary' (Review Title)
    """
    # Prefix for T5 models to signal the task
    inputs = ["summarize: " + doc for doc in examples["text"]]

    # Tokenize input text (Review content)
    model_inputs = tokenizer(
        inputs,
        max_length=config.max_source_len, # 512 as per CFG
        truncation=True,
        padding="max_length"
    )

    # Tokenize target text (Review Title)
    labels = tokenizer(
        text_target=examples["summary"],
        max_length=config.max_target_len, # 128 as per CFG
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the Amazon dataset
tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

# Remove the original text columns to save RAM
tokenized_dataset = tokenized_dataset.remove_columns(["text", "summary"])

print("Preprocessing complete.")
print(f"Train dataset size: {len(tokenized_dataset['train'])}")

# cell 5
# Rubric mapping: Generate Baseline (3)
from transformers import AutoModelForSeq2SeqLM, GenerationConfig
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading baseline model: {config.model_name} to {device}")

# We load in float32 for better compatibility with the fix
baseline_model = AutoModelForSeq2SeqLM.from_pretrained(
    config.model_name
).to(device)

# --- CRITICAL FIX FOR DYNAMICCACHE ERROR ---
baseline_model.config.use_cache = False
# -------------------------------------------

def generate_summaries(model, dataset, prompt_template, num_samples=20):
    """
    Generates summaries for a subset of the test data.
    """
    model.eval()
    results = []

    for i in range(num_samples):
        # Access original text from raw_dataset
        input_text = dataset[i]['text']
        reference = dataset[i]['summary']

        full_input = f"{prompt_template}\n{input_text}"

        inputs = tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_source_len
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                repetition_penalty=1.2,
                use_cache=False # Double-check cache is off during generation
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({
            "reference": reference,
            "generated": generated_text,
            "source": input_text[:200]
        })

    return pd.DataFrame(results)

print("✅ Fix applied. Baseline model and helper ready.")

# cell 6
# Rubric mapping: Generate Baseline (3)
import os

# 1. Define 3 standard/default baseline prompts as per PDF requirements
# These are simple, direct instructions without specific styling or constraints
baseline_prompt_templates = [
    "Summarize the following review:",
    "Provide a brief title for this text:",
    "TL;DR:"
]

# 2. Select the primary baseline prompt for the evaluation
# We use the first one as our 'Official Baseline'
primary_baseline_prompt = baseline_prompt_templates[0]

print(f"Running Baseline Generation using prompt: '{primary_baseline_prompt}'")

# 3. Generate summaries for 20 samples from the test set
# This ensures we have enough data for a meaningful initial evaluation
df_baseline = generate_summaries(
    model=baseline_model,
    dataset=raw_dataset['test'],
    prompt_template=primary_baseline_prompt,
    num_samples=20
)

# 4. Save results to /content/outputs as required by the TA-Engineer persona
save_path = f"{config.output_dir}/baseline_generations.csv"
df_baseline.to_csv(save_path, index=False)

print(f"\n✅ Baseline generation complete. Results saved to: {save_path}")

# 5. Display the first 5 results for qualitative inspection (Rubric: 'present representative outputs')
print("\n--- SAMPLE BASELINE OUTPUTS ---")
print(df_baseline[['reference', 'generated']].head(5))

# cell 8
# Rubric mapping: Quantitative Evaluation (6)
import evaluate
import numpy as np
import pandas as pd

# 1. Load the evaluation metrics
# ROUGE measures word overlap; BERTScore measures semantic meaning
# We install rouge_score here just in case it was missed earlier
!pip install -q rouge_score
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def calculate_metrics(df):
    """
    Calculates ROUGE and BERTScore.
    Uses .tolist() to prevent 'KeyError: 0' by bypassing Pandas indexing.
    """
    # Convert to standard Python lists for absolute safety
    preds = df['generated'].astype(str).tolist()
    refs = df['reference'].astype(str).tolist()

    print(f"Calculating ROUGE for {len(preds)} samples...")
    rouge_results = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    print("Calculating BERTScore (using roberta-large)...")
    bert_results = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang="en",
        model_type="roberta-large"
    )

    # Pack results into a dictionary
    return {
        "rouge1": round(rouge_results["rouge1"], 4),
        "rouge2": round(rouge_results["rouge2"], 4),
        "rougeL": round(rouge_results["rougeL"], 4),
        "bertscore_f1": round(np.mean(bert_results["f1"]), 4)
    }

# 2. Run initial evaluation on the baseline results from Cell 6
print("Running initial baseline evaluation...")
baseline_metrics = calculate_metrics(df_baseline)

# 3. Save to the metrics directory
metrics_df = pd.DataFrame([baseline_metrics])
metrics_save_path = f"{config.metrics_dir}/baseline_metrics.csv"
metrics_df.to_csv(metrics_save_path, index=False)

print("-" * 30)
print(f"✅ Baseline metrics saved to: {metrics_save_path}")
print(metrics_df.to_string(index=False))

# cell 9
# Rubric mapping: Prompt Engineering (5)
import pandas as pd
import os

# 1. Define Advanced Prompt Strategies
# We use 'text' as a placeholder for the few-shot template
prompt_strategies = {
    "Zero-Shot (Refined)": "Task: Summarize the following Amazon product review into a short, catchy headline of less than 10 words.\nReview:",
    "Few-Shot (2-Shot)": (
        "Review: I bought this for my daughter and she loves it! The color is vibrant and it fits perfectly.\nSummary: Great gift, perfect fit!\n\n"
        "Review: The item arrived broken and the customer service was unhelpful. Avoid this seller.\nSummary: Broken item, poor service.\n\n"
        "Review: {text}\nSummary:"
    )
}

# 2. Run Inference for 20 samples per strategy
all_prompt_data = []

for name, template in prompt_strategies.items():
    print(f"Testing Strategy: {name}...")
    for i in range(20):
        # Get raw data from the test set
        review_text = raw_dataset['test'][i]['text']
        true_summary = raw_dataset['test'][i]['summary']

        # Prepare the prompt
        if "{text}" in template:
            full_prompt = template.format(text=review_text)
        else:
            full_prompt = f"{template} {review_text}"


        # Tokenize and Generate
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            # ADDED use_cache=False HERE TO FIX THE ERROR
            outputs = baseline_model.generate(
                **inputs,
                max_new_tokens=30,
                use_cache=False
            )

# ... (rest of your code below remains the same)

        # Store results in a list of dictionaries
        all_prompt_data.append({
            "Strategy": name,
            "reference": true_summary,
            "generated": tokenizer.decode(outputs[0], skip_special_tokens=True)
        })

# 3. Create the Final Comparison Table
print("\nConsolidating all results...")
comparison_list = []

# A. Add the Baseline from Cell 7
# Use .copy() to ensure we don't mess up the original baseline variable
base_row = baseline_metrics.copy()
base_row['Strategy'] = "Original Baseline"
comparison_list.append(base_row)

# B. Calculate metrics for our new strategies
prompt_results_df = pd.DataFrame(all_prompt_data)

for name in prompt_strategies.keys():
    # Filter the data for just this one strategy
    subset = prompt_results_df[prompt_results_df['Strategy'] == name]
    # Call our fixed function from Cell 7
    scores = calculate_metrics(subset)
    scores['Strategy'] = name
    comparison_list.append(scores)

# 4. Save and Display
comparison_df = pd.DataFrame(comparison_list)
# Reorder columns to look good in the report
cols = ['Strategy', 'rouge1', 'rougeL', 'bertscore_f1']
comparison_df = comparison_df[cols]

os.makedirs(config.metrics_dir, exist_ok=True)
comparison_df.to_csv(f"{config.metrics_dir}/prompt_engineering_comparison.csv", index=False)

print("\n" + "="*50)
print("📊 TASK 5: PROMPT ENGINEERING COMPARISON")
print("="*50)
print(comparison_df.to_string(index=False))

# cell 10
# Rubric mapping: Fine-tuning (7)
import sys
import os
import torch

# 1. THE TRITON SHIELD:
# This stops bitsandbytes from trying to load the broken triton module
sys.modules["triton"] = None
sys.modules["triton.ops"] = None

from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

print("Preparing PEFT model with Triton-bypass...")

# 2. Re-initialize LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# 3. Force model to standard float32 to avoid needing specialized kernels
baseline_model.to(torch.float32)

# 4. Wrap the model
# Now that 'triton' is blocked in sys.modules, this should run normally
peft_model = get_peft_model(baseline_model, lora_config)

# 5. Setup Trainer
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

training_args = TrainingArguments(
    output_dir=config.output_dir,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    num_train_epochs=config.epochs,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

print("✅ SUCCESS! The Triton error has been bypassed.")
print(f"Trainable params: {sum(p.numel() for p in peft_model.parameters() if p.requires_grad)}")

# cell 11
# Rubric mapping: Fine-tuning (7)

print("Starting Fine-Tuning... This will take roughly 5-10 minutes.")

# 1. Start the training process
# You will see a progress bar and loss metrics appear here
trainer.train()

# 2. Save the adapter weights (the 'learned' part of the model)
peft_model_id = f"{config.model_save_dir}/amazon_summarizer_peft"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)

print(f"\n✅ Training complete! Fine-tuned weights saved to: {peft_model_id}")

# cell 12
# Rubric mapping: Quantitative Evaluation (6)
import torch
import gc
from tqdm import tqdm

print("🚀 Moving model to CPU for safe evaluation (Avoids OOM)...")

# 1. Move model to CPU and clear GPU
trainer.model.to("cpu")
torch.cuda.empty_cache()
gc.collect()

# 2. Manual Inference Loop (Very stable)
preds = []
labels = []

# We'll test on a subset of 100 samples to keep it fast on CPU
# This is plenty for a statistically significant ROUGE score comparison
test_subset = tokenized_dataset["test"].select(range(min(100, len(tokenized_dataset["test"]))))

print(f"Generating summaries for {len(test_subset)} samples...")
for example in tqdm(test_subset):
    # Prepare input
    input_ids = torch.tensor([example['input_ids']])

    # Generate
    with torch.no_grad():
        output_ids = trainer.model.generate(
            input_ids=input_ids,
            max_new_tokens=30,
            use_cache=False # Consistent with our fix from earlier
        )

    # Decode
    preds.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    labels.append(tokenizer.decode(example['labels'], skip_special_tokens=True))

# 3. Create DataFrame
df_finetuned = pd.DataFrame({
    'generated': preds,
    'reference': labels
})

# 4. Calculate Final Metrics
finetuned_metrics = calculate_metrics(df_finetuned)

# 5. Build and Display the Final Comparison Table
comparison_data = [
    {"Model": "Baseline (Pre-trained)", **baseline_metrics},
    {"Model": "Fine-Tuned (LoRA)", **finetuned_metrics}
]

final_comparison_df = pd.DataFrame(comparison_data)
final_comparison_df['Improvement (ROUGE-L)'] = (
    (final_comparison_df['rougeL'].pct_change().fillna(0) * 100).round(2).astype(str) + "%"
)

print("\n" + "="*50)
print("🏆 FINAL PERFORMANCE COMPARISON")
print("="*50)
print(final_comparison_df.to_string(index=False))

# Save results
final_comparison_df.to_csv(f"{config.metrics_dir}/final_model_comparison.csv", index=False)

# cell 13
# Rubric mapping: Qualitative Evaluation (6)

def generate_headline(review_text):
    # Move model back to CPU if it isn't already there to be safe
    trainer.model.to("cpu")

    inputs = tokenizer(
        f"Summarize: {review_text}",
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = trainer.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=30,
            use_cache=False
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test it out!
custom_review = "I've used this coffee maker for a week now. It brews fast and the thermal carafe keeps the coffee hot for hours. Highly recommend for the price!"
print(f"Custom Review: {custom_review}")
print(f"Generated Headline: {generate_headline(custom_review)}")
