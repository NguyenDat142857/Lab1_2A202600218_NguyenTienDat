import os
import time
from typing import Any, Callable
from openai import OpenAI

COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-4o": 0.010,
    "gpt-4o-mini": 0.0006,
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"


# -------------------- Task 1 --------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    end = time.time()

    text = response.choices[0].message.content
    latency = end - start

    return text, latency


# -------------------- Task 2 --------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:

    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


# -------------------- Task 3 --------------------
def compare_models(prompt: str) -> dict:

    gpt4o_resp, gpt4o_lat = call_openai(prompt)
    mini_resp, mini_lat = call_openai_mini(prompt)

    # ước lượng token (words → tokens)
    tokens_est = len(gpt4o_resp.split()) / 0.75
    cost = (tokens_est / 1000) * COST_PER_1K_OUTPUT_TOKENS["gpt-4o"]

    return {
        "gpt4o_response": gpt4o_resp,
        "mini_response": mini_resp,
        "gpt4o_latency": gpt4o_lat,
        "mini_latency": mini_lat,
        "gpt4o_cost_estimate": cost,
    }


# -------------------- Task 4 --------------------
def streaming_chatbot() -> None:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        history.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=history,
            stream=True,
        )

        print("Bot:", end=" ")

        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                print(content, end="", flush=True)
                reply += content

        print()

        history.append({"role": "assistant", "content": reply})

        # giữ 3 lượt hội thoại (6 messages)
        history = history[-6:]


# -------------------- Bonus A --------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(base_delay * (2 ** attempt))


# -------------------- Bonus B --------------------
def batch_compare(prompts: list[str]) -> list[dict]:

    results = []

    for p in prompts:
        res = compare_models(p)
        res["prompt"] = p
        results.append(res)

    return results


# -------------------- Bonus C --------------------
def format_comparison_table(results: list[dict]) -> str:

    def truncate(text):
        return text[:40] + "..." if len(text) > 40 else text

    table = "Prompt | GPT-4o | Mini | Latency4o | LatencyMini\n"
    table += "-" * 70 + "\n"

    for r in results:
        row = f"{truncate(r['prompt'])} | {truncate(r['gpt4o_response'])} | {truncate(r['mini_response'])} | {r['gpt4o_latency']:.2f} | {r['mini_latency']:.2f}"
        table += row + "\n"

    return table