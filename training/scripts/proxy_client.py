# proxy_client.py
"""
代理池客户端，对接 jhao104/proxy_pool 的 HTTP API。

API 文档参考: https://github.com/jhao104/proxy_pool
默认 API 地址: http://127.0.0.1:5010
"""

import time
from typing import Optional, List, Dict, Any

from curl_cffi import requests

# ==================== 配置 ====================
PROXY_POOL_API = "http://127.0.0.1:5010"

# 获取代理的重试次数
GET_RETRY = 3
# 单次 API 请求超时(秒)
API_TIMEOUT = 5
# 无可用代理时的等待时间(秒)
WAIT_WHEN_EMPTY = 10
# 是否在获取代理时验证其可用性(会额外消耗时间)
VALIDATE_BEFORE_USE = False
# ============================================

# 使用 curl_cffi 的 Session，避免与 Geeked 的 TLS 指纹冲突
_session = requests.Session(impersonate="chrome124")


# ---------------- 基础封装 ----------------

def _api_get(path: str, params: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """调用代理池 API，返回 JSON 或 None"""
    url = f"{PROXY_POOL_API}{path}"
    try:
        resp = _session.get(url, params=params, timeout=API_TIMEOUT)
        if resp.status_code != 200:
            print(f"[proxy_client] API {path} 返回 {resp.status_code}")
            return None
        return resp.json()
    except Exception as e:
        print(f"[proxy_client] API {path} 调用失败: {e}")
        return None


def _format_proxy(raw: str) -> Optional[str]:
    """把 ip:port 或带协议的字符串统一成 http://ip:port"""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith(("http://", "https://", "socks4://", "socks5://")):
        return raw
    return f"http://{raw}"


# ---------------- 核心接口 ----------------

def get_proxy(retry: int = GET_RETRY,
              proxy_type: Optional[str] = None) -> Optional[str]:
    """
    从代理池随机获取一个代理（不删除）。

    Args:
        retry: 失败重试次数
        proxy_type: 过滤代理协议，可选 "https"、"http"、"socks5"

    Returns:
        形如 "http://ip:port" 的字符串，失败返回 None
    """
    params = {}
    if proxy_type:
        params["type"] = proxy_type

    for attempt in range(retry):
        data = _api_get("/get/", params)
        if data and data.get("proxy"):
            proxy = _format_proxy(data["proxy"])
            if proxy:
                return proxy
        time.sleep(1)

    print("[proxy_client] 代理池无可用代理")
    return None


def pop_proxy(proxy_type: Optional[str] = None) -> Optional[str]:
    """
    获取并删除一个代理（不可复用）。
    适合一次性使用场景，避免代理被重复分配。
    """
    params = {}
    if proxy_type:
        params["type"] = proxy_type

    data = _api_get("/pop/", params)
    if data and data.get("proxy"):
        return _format_proxy(data["proxy"])
    return None


def delete_proxy(proxy: str) -> bool:
    """
    从代理池删除指定代理。
    用于标记失效代理，避免后续继续使用。
    """
    if not proxy:
        return False

    raw = proxy.replace("http://", "").replace("https://", "") \
               .replace("socks4://", "").replace("socks5://", "")

    data = _api_get("/delete/", {"proxy": raw})
    if data:
        print(f"[proxy_client] 已删除代理 {raw}")
        return True
    return False


def get_all(proxy_type: Optional[str] = None) -> List[str]:
    """获取所有代理"""
    params = {}
    if proxy_type:
        params["type"] = proxy_type

    data = _api_get("/all/", params)
    if data and isinstance(data, dict):
        proxies = data.get("proxy", [])
        return [_format_proxy(p) for p in proxies if p]
    return []


def get_count() -> int:
    """获取当前可用代理数量"""
    data = _api_get("/count/")
    if data and isinstance(data, dict):
        return data.get("count", 0)
    return 0


def wait_for_proxy(min_count: int = 1,
                   timeout: int = 300,
                   interval: int = 5) -> bool:
    """
    等待代理池积累到指定数量。
    适合采集脚本启动前预热。

    Args:
        min_count: 期望的最小代理数量
        timeout: 最长等待时间(秒)
        interval: 检查间隔(秒)

    Returns:
        True 表示达到数量，False 表示超时
    """
    start = time.time()
    while time.time() - start < timeout:
        count = get_count()
        if count >= min_count:
            print(f"[proxy_client] 代理池已就绪，可用 {count} 个")
            return True
        print(f"[proxy_client] 当前 {count} 个，等待达到 {min_count}...")
        time.sleep(interval)

    print(f"[proxy_client] 等待代理池超时 ({timeout}s)")
    return False


# ---------------- 代理可用性验证（可选） ----------------

def validate_proxy(proxy: str, test_url: str = "https://httpbin.org/ip",
                   timeout: int = 10) -> bool:
    """
    验证代理是否可用。
    默认关闭，因为每次验证会增加额外延迟。
    """
    try:
        resp = _session.get(test_url, proxies={"https": proxy, "http": proxy},
                            timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def get_valid_proxy(max_attempts: int = 5,
                    proxy_type: Optional[str] = None) -> Optional[str]:
    """
    获取一个经过验证的可用代理。
    如果代理池质量差，可以启用此函数替换 get_proxy()。
    """
    for _ in range(max_attempts):
        proxy = get_proxy(proxy_type=proxy_type)
        if not proxy:
            return None
        if validate_proxy(proxy):
            return proxy
        delete_proxy(proxy)
    return None


# ---------------- 自测 ----------------

if __name__ == "__main__":
    print("=== 代理池自测 ===")
    print(f"API 地址: {PROXY_POOL_API}")

    count = get_count()
    print(f"可用代理数量: {count}")

    if count == 0:
        print("代理池为空，请先启动代理池的 schedule 程序积累代理")
    else:
        p = get_proxy()
        print(f"随机获取代理: {p}")

        if p and VALIDATE_BEFORE_USE:
            ok = validate_proxy(p)
            print(f"代理可用性: {'可用' if ok else '不可用'}")