from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("gpt2")

dataset = load_dataset("imdb")

training_args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=1,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(500))
)

trainer.train()
