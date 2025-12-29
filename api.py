import requests
import logging
import json
import os
import httpx
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# 通过内部api调用DeepSeek时，会弹出InsecureRequestWarning，因此屏蔽该警告
urllib3.disable_warnings(InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CallBoyueModel():
    '''
    此类通过Boyue中转网站，调用Doubao、Gemini等模型
    '''
    def __init__(self, conf, api_key, base_url='http://35.220.164.252:3888/v1/', proxy_link: str | None = None):
        self.api_key = api_key
        self.base_url = base_url
        self.conf = conf
        self.model_name = self.conf['model_name']

        self.http_client = httpx.Client(proxy=proxy_link)
        # 将配置好的http客户端传递给OpenAI
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url=self.base_url,
            http_client=self.http_client
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, RateLimitError))
    )
    def get_response(self, system_prompt: str, user_prompt: str, think=True):
        api_params = {
                "model":self.model_name,
                "messages":[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature":self.conf.get('temperature', 0.7),
                "top_p":self.conf.get('top_p', 1.0),
                "max_tokens":self.conf.get('max_tokens', 512),
                "stream":  self.conf.get('stream',False)
        }

        if think is True:
            api_params['extra_body'] = {"thinking":{"type":"enabled"}}
        elif think is False:
            api_params['extra_body'] = {"thinking":{"type":"disabled"}}
        else:
            pass

        try:
            response = self.client.chat.completions.create(**api_params)
            if 'qwen' in self.model_name.lower():
                data = response.choices[0].message.content
            else:
                message = response.choices[0].message
                reasoning_content = getattr(message, 'reasoning_content', None) or getattr(message, 'reasoning', None)
                content = response.choices[0].message.content
                data = {
                    "reasoning_content":reasoning_content.strip() if reasoning_content else "",
                    "content":content.strip() if content else ""}
            return data
        except Exception as e:
            logging.error(f"{self.model_name} API failed after retry: {e}")
            return None, str(e)

    def close(self):
        self.http_client.close()

if __name__ == '__main__':
    proxy_url = "http://35.220.164.252:3888/v1"
    model_name = "doubao-seed-1-6-thinking-250615"
    api_key = "KEY"

    doubao_conf = {"model_name":model_name, 'temperature':0, "top_p":0.95, "max_tokens":5000, "stream":False}
    doubao = CallBoyueModel(conf=doubao_conf, api_key=api_key, proxy_link=proxy_url)

    response = doubao.get_response(system_prompt="", user_prompt="hello world", think=True)

    print(response)