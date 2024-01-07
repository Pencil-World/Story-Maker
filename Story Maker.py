# must print pipeline objects after calling object

def one():
    from transformers import pipeline

    generator = pipeline("text-generation")
    text = ""
    while True:
        text = (text + input())[-800:]
        text = generator(text, max_length=1_000, pad_token_id=50256, max_new_tokens=200)[0]["generated_text"]
        print(text)

"""
https://colab.research.google.com/drive/1SQmK0GYz34RGVlOnL5YMkdm7hXD6OjQT?usp=sharing#scrollTo=aNTmMJIMYjiC
!pip install transformers torch accelerate
!huggingface-cli login
https://huggingface.co/settings/tokens
New token, default settings
Add token as git credential? (Y/n) n
!huggingface-cli whoami
https://colab.research.google.com/drive/14loAculbJfT3DbeKkguZfjxFjYFjsfRS#scrollTo=uQIdE1LuLpcL
"""
def two():
    from transformers import AutoTokenizer, pipeline
    import torch

    print(torch.cuda.is_available())

    model = "meta-llama/Llama-2-7b-chat-hf" # meta-llama/Llama-2-7b-hf
    tokenizer = AutoTokenizer.from_pretrained(model)
    llama_pipeline = pipeline(
        "text-generation",  # LLM task
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    def get_llama_response(prompt: str) -> None:
        """
        Generate a response from the Llama model.

        Parameters:
            prompt (str): The user's input/question for the model.

        Returns:
            None: Prints the model's response.
        """
        sequences = llama_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=50,
        )
        print("Chatbot:", sequences[0]['generated_text'])

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "quit", "exit"]:
            print("Chatbot: Goodbye!")
            break
        get_llama_response(user_input)

one()
print("done")