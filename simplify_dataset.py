from datasets import load_dataset

# New optimized system message prompt text
optimized_system_message = (
    "As an assistant, you are required to answer questions using a fixed format. "
    "Your response must start with <|begin_of_thought|>, include your detailed thought process, and end with <|end_of_thought|>. "
)

def optimize_example(example):
    # Update the system field with the new optimized prompt
    example["system"] = optimized_system_message
    
    # Process each conversation message to update assistant messages
    new_conversations = []
    for conv in example.get("conversations", []):
        # For assistant messages, strip out solution markers
        if conv.get("from") == "assistant":
            text = conv.get("value", "")
            text = text.replace("<|begin_of_solution|>\n\n", "").replace("<|end_of_solution|>", "")
            conv["value"] = text
        new_conversations.append(conv)
    example["conversations"] = new_conversations
    return example

def main():
    # Load the dataset with the specified revision "main"
    dataset = load_dataset("dataset_stratos_17k", revision="main")
    
    # Use map to update both the system field and conversation content for each example
    updated_dataset = dataset.map(optimize_example)
    
    # Save the updated dataset to disk in a new folder
    updated_dataset.save_to_disk("optimized_dataset")

if __name__ == "__main__":
    main()