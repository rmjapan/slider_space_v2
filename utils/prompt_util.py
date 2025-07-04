import anthropic



# client = anthropic.Anthropic()
from typing import List, Optional

def claude_generate_prompts_sliders(prompt, 
                             num_prompts=20,
                             temperature=0.2, 
                             max_tokens=2000, 
                             frequency_penalty=0.0,
                             model="claude-3-5-sonnet-20240620",
                             verbose=False):
    assistant_prompt =  f''' You are an expert in writing diverse image captions. When i provide a prompt, I want you to give me {num_prompts} alternative prompts that is similar to the provided prompt but produces diverse images. Be creative and make sure the original subjects in the original prompt are present in your prompts. Make sure that you end the prompts with keywords that will produce high quality images like ",detailed, 8k" or ",hyper-realistic, 4k".

Give me the expanded prompts in the style of a list. start with a [ and end with ] do not add any special characters like \n 
I need you to give me only the python list and nothing else. Do not explain yourself

example output format:
["prompt1", "prompt2", ...]
'''
    
    user_prompt = prompt
    
    message=[
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
            ]
    
    output = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=assistant_prompt,
        messages=message
    )
    content = output.content[0].text
    return content




# def expand_prompts(concept_prompts: List[str], diverse_prompt_num: int, args) -> List[str]:
#     """
#     Expand the input prompts using Claude if requested.
    
#     Args:
#         concept_prompts: Initial list of prompts
#         diverse_prompt_num: Number of variations to generate per prompt
#         args: Training arguments
        
#     Returns:
#         List of expanded prompts
#     """
#     diverse_prompts = []
    
#     if diverse_prompt_num != 0:
#         for prompt in concept_prompts:
#             try:
#                 claude_generated_prompts = claude_generate_prompts_sliders(
#                     prompt=prompt,
#                     num_prompts=diverse_prompt_num,
#                     temperature=0.2,
#                     max_tokens=8000,
#                     frequency_penalty=0.0, 
#                     model="claude-3-5-sonnet-20240620",
#                     verbose=False
#                 )
#                 diverse_prompts.extend(eval(claude_generated_prompts))
#             except Exception as e:
#                 print(f"Error with Claude response: {e}")
#                 diverse_prompts.append(prompt)
#     else:
#         diverse_prompts = concept_prompts
        
#     print(f"Using prompts: {diverse_prompts}")
#     return diverse_prompts

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def gpt_generate_prompts_sliders(
        user_prompt,
        num_prompts=100,
        model="gpt-4.1",
        temperature=1,
        max_output_tokens=8192,
        top_p=1,
        store=True
                                     ):
    print("gpt")
    response = client.responses.create(
    model=model,
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",      
            "text":f"""You are an expert at generating anime-style character expression prompts. Your goal is to expand a provided base prompt into a diverse list of character expressions. Follow these exact guidelines:
                    1. Begin each prompt exactly with this base structure:
                    {user_prompt}

                    2. Add only emotional and expressive keywords to the end of each prompt to create subtle, rich, and varied facial expressions.

                    3. The emotional range should be expressed using only the following predefined tags: angry, joyful, sad, surprised, expressionless, embarrassed, blush, seductive smile, listless, moved, tears, scared, nervous, and smug. Each tag corresponds to a clearly defined facial expression. For example, 'joyful' may involve a bright smile or laughter, while 'scared' may be represented by wide-open eyes and trembling gestures. Use each tag based on its typical visual characteristics.

                    4. Use intensity modifiers (e.g., ':1.2', ':1.4') to emphasize certain emotional traits clearly.

                    5. Avoid repetition; ensure each prompt represents a distinct emotional nuance.

                    6. Do NOT include quality-enhancing terms (e.g.,"anime style","high resolution","4k")—only emotion-related keywords.

                    7. Generate exactly {num_prompts} unique prompts.

                    8. cutout ONLY a valid Python list of strings in This exact format:
                    ["prompt 1", "prompt 2", ..., "prompt 100"]
                    9. Limit each prompt strictly to 75 tokens or fewer.
                    """
            }
        
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "input_text",
            "text":user_prompt
            }
        ]
        }
    ],
    text={
        "format": {
        "type": "text"
        }
    },
    reasoning={},
    tools=[],
    temperature=1,
    max_output_tokens=max_output_tokens,
    top_p=1,
    store=True
    )
    print("GPT")
    return response.output_text

def clean_gpt_list_string(gpt_str):
    # 1. 先頭と末尾の空白を除去
    gpt_str = gpt_str.strip()
    
    # 2. 途中で途切れている場合の処理
    if not gpt_str.endswith(']'):
        # 最後の不完全な文字列を除去
        # 最後の完全な文字列の終わりを探す
        last_quote = gpt_str.rfind('"')
        if last_quote != -1:
            # 最後の引用符の後にある不完全な部分を除去
            gpt_str = gpt_str[:last_quote + 1]
        
        # 最後のカンマを探して除去
        last_comma = gpt_str.rfind(',')
        if last_comma != -1:
            gpt_str = gpt_str[:last_comma]
        
        # 3. 末尾に ] をつける
        gpt_str += ']'
    
    return gpt_str
def expand_prompts(concept_prompts: List[str], diverse_prompt_num: int, args) -> List[str]:
    """
    Expand the input prompts using Claude if requested.
    
    Args:
        concept_prompts: Initial list of prompts
        diverse_prompt_num: Number of variations to generate per prompt
        args: Training arguments
        
    Returns:
        List of expanded prompts
    """
    diverse_prompts = []
    print(f"diverse_prompt_num: {diverse_prompt_num}")
    
    if diverse_prompt_num != 0:
        for prompt in concept_prompts:
            try:
                gpt_generated_prompts = gpt_generate_prompts_sliders(
                    user_prompt=prompt,
                    num_prompts=diverse_prompt_num,
                    model="gpt-4.1-mini",
                    temperature=0.2,
                    max_output_tokens=5000,
                    top_p=1,
                    store=False
                )
                print(f"gpt_generated_prompts: {gpt_generated_prompts}")
                #"",でない最後の文字を削除
                gpt_generated_prompts = clean_gpt_list_string(gpt_generated_prompts)
                diverse_prompts.extend(eval(gpt_generated_prompts))
                
            except Exception as e:
                print(f"Error with GPT response: {e}")
                diverse_prompts.append(prompt)
            print("GUU")
        
            # try:
                # claude_generated_prompts = claude_generate_prompts_sliders(
            #         prompt=prompt,
            #         num_prompts=diverse_prompt_num,
            #         temperature=0.2,
            #         max_tokens=8000,
            #         frequency_penalty=0.0, 
            #         model="claude-3-5-sonnet-20240620",
            #         verbose=False
            #     )
            #     diverse_prompts.extend(eval(claude_generated_prompts))
            # except Exception as e:
            #     print(f"Error with Claude response: {e}")
            #     diverse_prompts.apped(prompt)
    else:
        diverse_prompts = concept_prompts
        
    print(f"Using prompts: {diverse_prompts}")
    return diverse_prompts


import re
import numpy as np

def extract_emotions(prompt):
    """
    入力プロンプトからベース部分（generalまで）と感情タグを抽出
    """
    tags = prompt.strip().strip('"').split(",")
    idx = tags.index("general") + 1 if "general" in tags else len(tags)
    base_tags = tags[:idx]
    emotion_tags = tags[idx:]

    parsed_emotions = []
    for tag in emotion_tags:
        match = re.match(r"(.+?):([\d.]+)", tag)
        if match:
            parsed_emotions.append(match.group(1))
        else:
            parsed_emotions.append(tag)
    return base_tags, parsed_emotions

def generate_individual_strength_variations(prompt, batch_size, min_val=0.8, max_val=1.6, step=0.1, seed=None):
    """
    感情ごとに個別の強度（0.1刻み）を設定してバッチ分のプロンプトを生成
    """
    if seed is not None:
        np.random.seed(seed)

    base_tags, emotions = extract_emotions(prompt)
    possible_strengths = np.round(np.arange(min_val, max_val + step, step), 1)
    variations = []

    for _ in range(batch_size):
        emotion_strs = []
        for e in emotions:
            val = np.random.choice(possible_strengths)
            emotion_strs.append(f"{e}:{val:.1f}")
        full_prompt = ",".join(base_tags + emotion_strs)
        variations.append(full_prompt)

    return variations

# # ---------- 実行例 ----------
# if __name__ == "__main__":
#     prompt = "1 girl,solo,masterpiece,hatsune miku,best quality,character looking straight ahead,front view,head shot,simple background,general,nervous:1.4,embarrassed:1.3"
#     batch_size = 5

#     prompts = generate_individual_strength_variations(prompt, batch_size, seed=42)

#     for i, p in enumerate(prompts, 1):
#         print(f"[Prompt {i}]\n{p}\n")

