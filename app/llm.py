import os

def llm_answer(system: str, user: str) -> str | None:
    api_key = None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=0.1,
            max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Fail soft to context-only if anything goes wrong
        return None
