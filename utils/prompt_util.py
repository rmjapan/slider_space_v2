import anthropic



client = anthropic.Anthropic()
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
    
    if diverse_prompt_num != 0:
        for prompt in concept_prompts:
            try:
                claude_generated_prompts = claude_generate_prompts_sliders(
                    prompt=prompt,
                    num_prompts=diverse_prompt_num,
                    temperature=0.2,
                    max_tokens=8000,
                    frequency_penalty=0.0, 
                    model="claude-3-5-sonnet-20240620",
                    verbose=False
                )
                diverse_prompts.extend(eval(claude_generated_prompts))
            except Exception as e:
                print(f"Error with Claude response: {e}")
                diverse_prompts.append(prompt)
    else:
        diverse_prompts = concept_prompts
        
    print(f"Using prompts: {diverse_prompts}")
    return diverse_prompts

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

    response = client.responses.create(
    model=model,
    input=[
        {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            # "text": " You are an expert in writing diverse image captions. When i provide a prompt, I want you to give me {num_prompts} alternative prompts that is similar to the provided prompt but produces diverse images. Be creative and make sure the original subjects in the original prompt are present in your prompts. Make sure that you end the prompts with keywords that will produce high quality images like \",detailed, 8k\" or \",hyper-realistic, 4k\".\n\nGive me the expanded prompts in the style of a list. start with a [ and end with ] do not add any special characters like \\n \nI need you to give me only the python list and nothing else. Do not explain yourself\n\nexample output format:\n[\"prompt1\", \"prompt2\", ...]\n"
            # }
            # "text": f"You are an expert at crafting anime-style portrait prompts. The concept prompt: I give you (e.g., “a kind-looking man in his 20s”, “a studious-looking high-school girl”) is an abstract impression only. To translate that impression into vivid, diverse visual prompts, generate {num_prompts} English prompts that obey all rules below:\n\n1. Keep the face front-and-center in every prompt.\n2. Vary the expression keyword to match the impression (e.g., shy glance, confident gaze, hopeful eyes).\n3. Vary the expression intensity (e.g., slightly smiling, deeply caring) to add gradation.\n4. Limit background and props to the bare minimum; only subtle gestures that support the impression are allowed.\n5. Explicitly state the anime style (include “anime style”, “anime portrait”, etc.).\n6. Append a fixed quality footer such as “, anime style, ultra-detailed, 4k”.\n7. Enrich each prompt with similes, metaphors, or hyperbole to concretize the abstract concept (e.g., a smile like a spring breeze, eyes brighter than dawn).\n8. Do not repeat identical phrases across prompts; each must convey a unique emotional nuance.\n9. Return only a valid Python list in this exact format—no explanations, no extra characters, no newlines: [\"prompt 1\", \"prompt 2\", ...]",
            # }
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
            # # "text": f"""You are an expert at crafting anime-style portrait prompts. I will give you an abstract impression concept (e.g., “a kind-looking man in his 20s”, “a studious-looking high-school girl”), and your task is to translate that into vivid, diverse visual prompts that obey all the following rules:\n\n1. Center the face prominently in every prompt.\n2. Interpret the given impression across a wide emotional spectrum—not only joy, anger, sorrow, or pleasure, but also subtler or mixed states like serenity, unease, awe, melancholy, or whimsy.\n3. Vary the expression with emotionally evocative keywords (e.g., a wistful glance, a piercing gaze, a dreamy half-smile).\n4. Adjust intensity levels (e.g., barely perceptible smile, burning determination) to provide fine-grained emotional variation.\n5. Keep background and props minimal; only light gestures or contextual hints are allowed to support the impression.\n6. Always include explicit anime-style descriptors such as “anime style”, “anime portrait”, etc.\n7. Append a quality-enhancing footer: “, anime portrait, ultra-detailed, high resolution, 4k”.\n8. Integrate vivid figurative language (similes, metaphors, hyperbole) to make each prompt uniquely expressive (e.g., 'eyes like stormy skies', 'a smile as faint as fading starlight').\n9. Avoid phrase repetition. Every prompt must express a distinct emotional nuance.\n10. Return a valid Python list in the exact format: \["prompt 1", "prompt 2", ...] with no extra characters, no newlines, no explanations."""
            # }


            
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
    max_output_tokens=2048,
    top_p=1,
    store=True
    )
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
                    store=True
                )
                print(f"gpt_generated_prompts: {gpt_generated_prompts}")
                #"",でない最後の文字を削除
                gpt_generated_prompts = clean_gpt_list_string(gpt_generated_prompts)
                diverse_prompts.extend(eval(gpt_generated_prompts))
                
            except Exception as e:
                print(f"Error with GPT response: {e}")
                diverse_prompts.append(prompt)
        
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

