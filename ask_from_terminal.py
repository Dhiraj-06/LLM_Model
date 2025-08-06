import requests
import json
import os

# --- NEW: Load the API Key from Environment Variables ---
API_KEY = os.environ.get("MY_API_KEY")

# The URL of your running API server
API_URL = "http://127.0.0.1:5000/ask"

def main():
    if not API_KEY:
        print("FATAL ERROR: The MY_API_KEY environment variable is not set.")
        print("Please set it before running this client.")
        return

    print("--- RAG API Terminal Client ---")
    print("Type your question and press Enter. Type 'exit' to quit.")

    while True:
        question = input("\nQuestion: ")
        if question.lower().strip() == 'exit':
            break
        
        # --- NEW: Prepare headers with the API Key ---
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": API_KEY  # This sends your secret key
        }
        payload = {"question": question}

        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 401:
                print("❌ ERROR: Unauthorized. Your API key may be incorrect.")
            elif response.status_code == 200:
                answer_data = response.json()
                print(f"✅ Answer: {answer_data.get('answer', 'No answer key found.')}")
            else:
                print(f"❌ ERROR: Received status code {response.status_code}")
                print(response.text)

        except requests.exceptions.RequestException:
            print(f"\n❌ ERROR: Could not connect to the API server at {API_URL}.")
            break

if __name__ == "__main__":
    main()