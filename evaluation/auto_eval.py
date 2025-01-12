import argparse
import os
import json
import time
import re
import base64
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""
USER_PROMPT = """TASK: <task>
Result Response: <answer>
<num> screenshots at the end: """


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def auto_eval_by_model(process_dir, client, model_type, api_model, img_num):
    print(f'--------------------- {process_dir} ---------------------')
    res_files = sorted(os.listdir(process_dir))
    with open(os.path.join(process_dir, 'interact_messages.json')) as fr:
        it_messages = json.load(fr)

    if len(it_messages) == 1:
        print('Not find answer for ' + process_dir + ' only system messages')
        print()
        return 0

    task_info = it_messages[1]["content"]
    if type(task_info) == list:
        task_info = task_info[0]["text"]
    assert 'Now given a task' in task_info
    pattern = r"Now given a task:(.+?)Please interact with"
    matches = re.search(pattern, task_info)
    task_content = matches.group(1).strip()

    ans_info = it_messages[-1]["content"]
    if 'Action: ANSWER' not in ans_info:
        print('Not find answer for ' + process_dir)
        print()
        return 0
    pattern_ans = r"ANSWER[; ]+\[?(.[^\]]*)\]?"
    matches_ans = re.search(pattern_ans, ans_info)
    answer_content = matches_ans.group(1).strip()

    # max_screenshot_id = max([int(f[10:].split('.png')[0]) for f in os.listdir(process_dir) if '.png' in f])
    # final_screenshot = f'screenshot{max_screenshot_id}.png'
    # b64_img = encode_image(os.path.join(process_dir, final_screenshot))
    whole_content_img = []
    pattern_png = r'screenshot(\d+)\.png'
    matches = [(filename, int(re.search(pattern_png, filename).group(1))) for filename in res_files if re.search(pattern_png, filename)]
    matches.sort(key=lambda x: x[1])
    end_files = matches[-img_num:]
    for png_file in end_files:
        b64_img = encode_image(os.path.join(process_dir, png_file[0]))
        whole_content_img.append(
            {
                'type': 'image_url',
                'image_url': {"url": f"data:image/png;base64,{b64_img}"}
            }
        )

    user_prompt_tmp = USER_PROMPT.replace('<task>', task_content)
    user_prompt_tmp = user_prompt_tmp.replace('<answer>', answer_content)
    user_prompt_tmp = user_prompt_tmp.replace('<num>', str(img_num))
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_prompt_tmp}
            ]
            + whole_content_img
            + [{'type': 'text', 'text': "Your verdict:\n"}]
        }
    ]
    while True:
        try:
            print(f'Calling {model_type} API to get the auto evaluation......')
            if model_type == "openai":
                openai_response = client.chat.completions.create(
                    model=api_model, messages=messages, max_tokens=1000, seed=42, temperature=0
                )
                response_content = openai_response.choices[0].message.content
                tokens = {
                    'prompt': openai_response.usage.prompt_tokens,
                    'completion': openai_response.usage.completion_tokens
                }
            elif model_type == "gemini":
                try:
                    content_parts = [Part.from_text(user_prompt_tmp)]
                    
                    # Add images
                    for png_file in end_files:
                        with open(os.path.join(process_dir, png_file[0]), "rb") as f:
                            image_data = f.read()
                            content_parts.append(
                                Part.from_data(data=image_data, mime_type="image/png")
                            )

                    # Generate content with retry logic
                    retry_count = 0
                    max_retries = 3
                    while retry_count < max_retries:
                        try:
                            response = client.generate_content(
                                content_parts,
                                generation_config={
                                    "max_output_tokens": 1000,
                                    "temperature": 0,
                                },
                                safety_settings=[
                                    SafetySetting(
                                        category=category,
                                        threshold=SafetySetting.HarmBlockThreshold.OFF
                                    )
                                    for category in [
                                        SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                        SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                        SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                        SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                    ]
                                ]
                            )
                            return response.text
                        except Exception as e:
                            print(f"Error calling Gemini API (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                            if "NotFound" in str(e):
                                # If the model is not found, try with a different model version
                                if client.model_name == "gemini-1.5-pro-vision":
                                    print("Falling back to gemini-pro-vision")
                                    client = GenerativeModel("gemini-pro-vision")
                                else:
                                    print("Both model versions failed")
                                    return None
                            retry_count += 1
                            if retry_count < max_retries:
                                time.sleep(10)
                            else:
                                print("Max retries reached")
                                return None

                except Exception as e:
                    print(f"Error in content preparation: {str(e)}")
                    time.sleep(10)
                    return None

            print('Prompt Tokens:', tokens['prompt'], ';',
                  'Completion Tokens:', tokens['completion'])
            print('API call complete...')
            break
        except Exception as e:
            print(e)
            if type(e).__name__ == 'RateLimitError':
                time.sleep(10)
            elif type(e).__name__ == 'APIError':
                time.sleep(15)
            elif type(e).__name__ == 'InvalidRequestError':
                exit(0)
            else:
                time.sleep(10)
    return response_content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_dir', type=str, default='results')
    parser.add_argument('--lesson_dir', type=str, default='results')
    parser.add_argument("--api_key", default=None, type=str, 
                       help="OpenAI API key (only needed for OpenAI models)")
    parser.add_argument("--api_model", default="gpt-4-vision-preview", type=str, help="api model name")
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--model_type", default="openai", choices=["openai", "gemini"],
                       help="Model provider to use")
    parser.add_argument("--gcp_project", default="testing-vertex-ai-438519", 
                       help="Google Cloud project ID")
    parser.add_argument("--gcp_location", default="us-central1", 
                       help="Google Cloud location")
    args = parser.parse_args()

    if args.model_type == "openai":
        client = OpenAI(api_key=args.api_key)
    elif args.model_type == "gemini":
        try:
            # Try to get default credentials
            credentials, project = default()
            
            # Initialize Vertex AI
            vertexai.init(
                project=args.gcp_project,
                location=args.gcp_location,
                credentials=credentials  # Use proper credentials object
            )
            
            # Create model
            client = GenerativeModel("gemini-pro-vision")
            
        except DefaultCredentialsError as e:
            print(f"Error with Google Cloud credentials: {str(e)}")
            print("Please ensure you have authenticated with 'gcloud auth application-default login'")
            exit(1)

    webs = ['Allrecipes', 'Amazon', 'Apple', 'ArXiv', 'BBC News', 'Booking', 'Cambridge Dictionary',
            'Coursera', 'ESPN', 'GitHub', 'Google Flights', 'Google Map', 'Google Search', 'Huggingface', 'Wolfram Alpha']

    for web in webs:
        web_task_res = []
        for idx in range(0, 46):
            file_dir = os.path.join(args.process_dir, 'task'+web+'--'+str(idx))
            if os.path.exists(file_dir):
                response = auto_eval_by_model(file_dir, client, args.model_type, args.api_model, args.max_attached_imgs)
                web_task_res.append(response)
            else:
                pass
        if web_task_res:
            print(web_task_res)
if __name__ == '__main__':
    main()
