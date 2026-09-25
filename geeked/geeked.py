# geeked/geeked.py
from uuid import uuid4
from curl_cffi import requests
import random, time, json
from geeked.sign import Signer


class Geeked:
    # ============ 延迟配置 ============
    REQUEST_DELAY_RANGE = (0.5, 1.0)   # 每次远程请求后的延迟范围(秒)
    # ==================================

    def __init__(self, captcha_id: str, captcha_type: str = None,
                 risk_type: str = None, proxy: str = None, **kwargs):
        self.pass_token = None
        self.lot_number = None
        self.captcha_id = captcha_id
        self.challenge = str(uuid4())
        # 兼容两种字段命名
        self.captcha_type = captcha_type or risk_type
        self.risk_type = self.captcha_type    # 兼容原有属性访问
        self.callback = Geeked.random()

        session_kwargs = {"impersonate": "chrome124", **kwargs}
        if proxy:
            session_kwargs["proxies"] = {"https": proxy, "http": proxy}
        self.session = requests.Session(**session_kwargs)

        self.session.headers = {
            "connection": "keep-alive",
            "sec-ch-ua-platform": "\"Windows\"",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "sec-ch-ua-mobile": "?0",
            "accept": "*/*",
            "sec-fetch-site": "same-origin",
            "sec-fetch-mode": "no-cors",
            "sec-fetch-dest": "script",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9"
        }
        self.session.base_url = "https://gcaptcha4.geevisit.com"

    @staticmethod
    def random() -> str:
        return f"geetest_{int(random.random() * 10000) + int(time.time() * 1000)}"

    def _delay(self):
        """每次远程请求后的统一延迟"""
        delay = random.uniform(*self.REQUEST_DELAY_RANGE)
        time.sleep(delay)

    def set_proxy(self, proxy: str):
        if proxy:
            self.session.proxies = {"https": proxy, "http": proxy}
        else:
            self.session.proxies = {}

    def reset_challenge(self):
        self.challenge = str(uuid4())
        self.callback = Geeked.random()

    def format_response(self, response: str) -> dict:
        if f"{self.callback}(" not in response:
            raise ValueError(f"响应不含 callback，实际内容: {response[:200]}")
        return json.loads(response.split(f"{self.callback}(")[1][:-1])["data"]

    def load_captcha(self):
        self.reset_challenge()

        params = {
            "captcha_id": self.captcha_id,
            "challenge": self.challenge,
            "client_type": "web",
            "risk_type": self.captcha_type,
            "lang": "zho",
            "callback": self.callback,
        }
        res = self.session.get("/load", params=params)

        # 请求后延迟
        self._delay()

        return self.format_response(res.text)

    def submit_captcha(self, data: dict) -> dict:
        self.callback = Geeked.random()

        params = {
            "callback": self.callback,
            "captcha_id": self.captcha_id,
            "client_type": "web",
            "lot_number": self.lot_number,
            "risk_type": self.captcha_type,
            "payload": data["payload"],
            "process_token": data["process_token"],
            "payload_protocol": "1",
            "pt": "1",
            "w": Signer.generate_w(data, self.captcha_id, self.captcha_type),
        }
        res = self.session.get("/verify", params=params).text

        # 请求后延迟
        self._delay()

        res = self.format_response(res)

        if res.get("seccode") is None:
            raise Exception(f"Failed to submit captcha: {res}")

        return res["seccode"]

    def download(self, url: str, timeout: int = 10):
        """封装图片下载，自动加延迟"""
        resp = self.session.get(url, timeout=timeout)
        self._delay()
        return resp

    def solve(self) -> dict:
        data = self.load_captcha()
        self.lot_number = data["lot_number"]
        seccode = self.submit_captcha(data)
        return seccode