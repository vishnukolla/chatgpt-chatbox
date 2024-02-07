from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ChatGPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_response(self, user_input, max_length=100):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        response_ids = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

# Initialize the chatbot
chatbot = ChatGPT()

# Example conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("ChatGPT: Goodbye!")
        break
    response = chatbot.generate_response(user_input)
    print("ChatGPT:", response)
