# finetune.py

from transformers import TrainingArguments
from intel_extension_for_transformers.neural_chat.config import (
    ModelArguments,
    DataArguments,
    FinetuningArguments,
    TextGenerationFinetuningConfig,
)
from intel_extension_for_transformers.neural_chat.chatbot import finetune_model

def main():
    # Define model arguments
    model_args = ModelArguments(model_name_or_path="meta-llama/Llama-2-7b-chat-hf")

    # Define data arguments (adjust paths and parameters as needed)
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

if __name__ == "__main__":
    main()
