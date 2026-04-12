import asyncio
import aiohttp


async def create_completion(session, prompt, model="gpt-4o-mini-2024-07-18"):
    API_KEY = 'sk-xWnLMDe1yLYY7A7f1Q3gA8uokKlPfe8DomakThnJ15W391ok'
    BASE_URL = 'https://yunwu.ai/v1'
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    retry = 0
    while retry < 10:
        try:
            async with session.post(url=f"{BASE_URL}/chat/completions",
                json={
                    "model": model,
                    "max_tokens": 4000,
                    #"max_tokens": 500,
                    "temperature": 0,
                    # "frequency_penalty": 0.05,
                    # "presence_penalty": 0.0,
                    # "top_p": 1,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(result['choices'][0]['message']['content'])
                    print('-----------------------')
                    return result['choices'][0]['message']['content']
                else:
                    error_text = await response.text()
                    print(f"请求失败，状态码: {response.status}, 错误信息: {error_text}")
                    retry += 1
                    if retry < 10:
                        await asyncio.sleep(1)
                    else:
                        return f"[API Error: {response.status}]"
        except Exception as e:
            retry += 1
            if retry < 10:
                await asyncio.sleep(1)
                print(f"Error: {e}, 重试 {retry}/10", flush=True)
            else:
                print(f"Error after 10 retries: {e}", flush=True)
                return f"[Exception: {str(e)}]"

async def run_async(prompts, model="gpt-4o-mini-2024-07-18", max_concurrent=2000):
    """
    异步批量处理 prompts（串行执行，max_concurrent=1）
    
    Args:
        prompts: 待处理的 prompt 列表
        model: 模型名称
        max_concurrent: 最大并发数，默认1（串行）
    
    Returns:
        responses: 响应内容列表
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def create_completion_with_limit(session, prompt, model):
        async with semaphore:
            return await create_completion(session, prompt, model)
    
    async with aiohttp.ClientSession() as session:
        tasks = [create_completion_with_limit(session, prompt, model) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
    
    return responses

if __name__ == "__main__":
    prompts = ['who are you?', 'what you like?']
    responses = asyncio.run(run_async(prompts))
    print('--------')
    print(responses)