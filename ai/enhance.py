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

def process_single_item(chains_to_try, item: Dict, language: str) -> Dict:
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

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }

    import json
    import re

    # 尝试使用不同的链直到成功
    last_exception = None
    for i, chain in enumerate(chains_to_try):
        try:
            response = chain.invoke({
                "language": language,
                "content": item['summary']
            })

            # 响应现在是字符串，需要从中提取JSON
            response_str = str(response)

            # 调试：打印响应内容的前缀
            print(f"Response for {item.get('id', 'unknown')} chain {i+1}: {response_str[:100]}...", file=sys.stderr)

            # 尝试从响应中提取JSON对象
            # 查找第一个 { 和最后一个 } 之间的内容
            start_idx = response_str.find('{')
            end_idx = response_str.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = response_str[start_idx:end_idx+1]
                try:
                    ai_result = json.loads(json_str)

                    # 确保所有必需字段都存在
                    for field in default_ai_fields.keys():
                        if field not in ai_result:
                            ai_result[field] = default_ai_fields[field]

                    item['AI'] = ai_result
                    print(f"Successfully parsed JSON for {item.get('id', 'unknown')}", file=sys.stderr)
                    break  # 成功后跳出循环
                except json.JSONDecodeError as json_err:
                    print(f"JSON parsing failed for {item.get('id', 'unknown')}: {json_err}", file=sys.stderr)
                    print(f"Trying alternative parsing for response: {response_str[:200]}...", file=sys.stderr)

                    # 尝试替代解析方法：查找可能的JSON模式
                    # 使用正则表达式查找键值对
                    json_pattern = r'"([^"]+)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                    matches = re.findall(json_pattern, response_str)

                    if len(matches) >= 3:  # 至少有3个键值对，可能是一个完整的结构
                        # 尝试重建JSON结构
                        ai_result = {}
                        for key, value in matches:
                            # 清理键名，移除可能的引号或其他字符
                            clean_key = key.strip().strip('"').lower()
                            if clean_key in default_ai_fields:
                                ai_result[clean_key] = value

                        # 确保所有必需字段都存在
                        for field in default_ai_fields.keys():
                            if field not in ai_result:
                                ai_result[field] = default_ai_fields[field]

                        item['AI'] = ai_result
                        print(f"Successfully parsed with regex for {item.get('id', 'unknown')}", file=sys.stderr)
                        break  # 成功后跳出循环
                    else:
                        print(f"Regex parsing also failed for {item.get('id', 'unknown')}", file=sys.stderr)
                        continue  # 尝试下一个链
            else:
                # 如果找不到JSON格式，尝试查找可能的JSON模式
                # 使用正则表达式查找键值对
                json_pattern = r'"([^"]+)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                matches = re.findall(json_pattern, response_str)

                if len(matches) >= 3:  # 至少有3个值，可能是一个完整的结构
                    # 尝试重建JSON结构
                    ai_result = {}
                    for key, value in matches:
                        # 清理键名，移除可能的引号或其他字符
                        clean_key = key.strip().strip('"').lower()
                        if clean_key in default_ai_fields:
                            ai_result[clean_key] = value

                    # 确保所有必需字段都存在
                    for field in default_ai_fields.keys():
                        if field not in ai_result:
                            ai_result[field] = default_ai_fields[field]

                    item['AI'] = ai_result
                    print(f"Successfully parsed with regex (no JSON delimiters) for {item.get('id', 'unknown')}", file=sys.stderr)
                    break  # 成功后跳出循环
                else:
                    print(f"No JSON-like structure found in response for {item.get('id', 'unknown')}", file=sys.stderr)
                    continue  # 尝试下一个链

        except Exception as e:
            last_exception = e
            print(f"Chain {i+1} failed for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
            continue  # 尝试下一个链

    # 如果所有链都失败，使用默认值
    if 'AI' not in item:
        print(f"All chains failed for {item.get('id', 'unknown')}, using default values: {last_exception}", file=sys.stderr)
        item['AI'] = default_ai_fields

    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

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
            executor.submit(process_single_item, [chain], item, language): idx
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