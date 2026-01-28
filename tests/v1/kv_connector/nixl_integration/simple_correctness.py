import openai

# Configuration
BASE_URL = "http://localhost:8192/v1"
MODEL_NAME = "/model/"
MODEL_NAME="deepseek-ai/DeepSeek-V2-Lite"
PROMPT = "Typical Chinese breakfast includes "

def run_simple_test():
    print(f"Connecting to {BASE_URL}...")
    client = openai.OpenAI(api_key="EMPTY", base_url=BASE_URL)
    
    print(f"Sending prompt: '{PROMPT}'")
    try:
        completion = client.completions.create(
            model=MODEL_NAME,
            prompt=PROMPT,
            max_tokens=30,
            temperature=0
        )
        
        output_text = completion.choices[0].text
        print("\n" + "="*50)
        print("OUTPUT RESULT:")
        print(f"Prompt: {PROMPT}")
        print(f"Generated: {output_text}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error during request: {e}")

if __name__ == "__main__":
    run_simple_test()
