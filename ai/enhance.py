import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
import os
template_path = os.path.join(os.path.dirname(__file__), "template.txt")
system_path = os.path.join(os.path.dirname(__file__), "system.txt")
template = open(template_path, "r").read()
system = open(system_path, "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


REQUIRED_AI_FIELDS = (
    "tldr",
    "motivation",
    "method",
    "result",
    "conclusion",
)

DEFAULT_AI_FIELDS = {
    "tldr": "Summary generation failed",
    "motivation": "Motivation analysis unavailable",
    "method": "Method extraction failed",
    "result": "Result analysis unavailable",
    "conclusion": "Conclusion extraction failed",
}


def parse_ai_response(response) -> Dict[str, str]:
    response_text = str(response)
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}")
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        raise ValueError("Response does not contain a JSON object")

    try:
        result = json.loads(response_text[start_idx:end_idx + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response contains invalid JSON: {exc}") from exc

    if not isinstance(result, dict):
        raise ValueError("Response JSON must be an object")

    for field in REQUIRED_AI_FIELDS:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Required field '{field}' is missing or empty")

    return result


MAX_AI_RETRIES = 30
MAX_RETRY_DELAY_SECONDS = 2


def generate_ai_fields(
    chain,
    inputs: Dict[str, str],
    paper_id: str,
    max_retries: int = MAX_AI_RETRIES,
    sleep_fn=time.sleep,
) -> Dict[str, str]:
    total_attempts = max_retries + 1
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            response = chain.invoke(inputs)
            result = parse_ai_response(response)
            print(
                f"Summary generated for {paper_id} on attempt "
                f"{attempt}/{total_attempts}",
                file=sys.stderr,
            )
            return result
        except Exception as exc:
            last_error = exc
            print(
                f"Summary attempt {attempt}/{total_attempts} failed for "
                f"{paper_id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

        if attempt < total_attempts:
            retry_number = attempt
            delay = min(2 ** (retry_number - 1), MAX_RETRY_DELAY_SECONDS)
            sleep_fn(delay)

    print(
        f"All {total_attempts} summary attempts failed for {paper_id}; "
        f"using default values: {last_error}",
        file=sys.stderr,
    )
    return DEFAULT_AI_FIELDS.copy()


def process_single_item(chain, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", False)
            else:
                # 如果接口异常，默认不触发敏感词
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            # 当连接失败时，假设内容不敏感，避免所有内容都被过滤
            return False

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    item["AI"] = generate_ai_fields(
        chain,
        {
            "language": language,
            "content": item["summary"],
        },
        item.get("id", "unknown"),
    )

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None

    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    # 创建基础模型
    llm = ChatOpenAI(model=model_name)

    # 创建一个强制输出JSON格式的提示
    json_system = system + "\n\nIMPORTANT: Respond in valid JSON format with the following structure:\n" + \
                 '{{"tldr": "...", "motivation": "...", "method": "...", "result": "...", "conclusion": "..."}}\n' + \
                 'Ensure your entire response is a single, valid JSON object with no additional text before or after.'

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(json_system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    # 创建一个简单的链，直接输出文本，然后手动解析为JSON
    from langchain_core.output_parsers import StrOutputParser
    chain = prompt_template | llm | StrOutputParser()

    print('Connect to:', model_name, file=sys.stderr)

    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，传递chains_to_try列表而不是单个chain
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }

        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                if data[idx] is not None:
                    processed_data[idx] = data[idx].copy()  # 复制原始数据
                    processed_data[idx]['AI'] = {
                        "tldr": "Processing failed",
                        "motivation": "Processing failed",
                        "method": "Processing failed",
                        "result": "Processing failed",
                        "conclusion": "Processing failed"
                    }
                else:
                    processed_data[idx] = None

    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )

    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
