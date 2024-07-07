from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Fine-tuning

# Define model arguments
model_args = ModelArguments(model_name_or_path="meta-llama/Llama-2-7b-chat-hf")

# Define data arguments (example with a JSON file)
data_args = DataArguments(train_file="alpaca_data.json", validation_split_percentage=1)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./tmp',
    do_train=True,
    do_eval=True,
    num_train_epochs=3,
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    save_strategy="no",
    log_level="info",
    save_total_limit=2,
    bf16=True,
)

# Define fine-tuning arguments
finetune_args = FinetuningArguments()

# Combine into a fine-tuning config
finetune_cfg = TextGenerationFinetuningConfig(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
    finetune_args=finetune_args,
)

# Perform fine-tuning
finetune_model(finetune_cfg)

# Step 2: Inference and Interaction

# Load the fine-tuned model and tokenizer
model_name = "./tmp"  # Path to your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate responses
def generate_response(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Chatbot interaction loop
def chat():
    print("Start chatting with the bot (type 'exit' to stop)!")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
