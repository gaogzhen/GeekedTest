# fetch_bg.py
import os
import re
import sys
import time
import random
import signal
import atexit
from geeked.geeked import Geeked
from proxy_client import get_proxy, delete_proxy, validate_proxy

# ==================== 配置 ====================
CAPTCHA_ID = "af29b3003fc94f2ba29e865b31ee86ee"
CAPTCHA_TYPE = "icon"                    # 字段名改为 captcha_type
SAVE_DIR = "dataset/images/all"
TOTAL = 2010
STATIC_BASE = "https://static.geetest.com"

SLEEP_RANGE = (2, 4)                      # 成功后的额外间隔
MAX_RETRY = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 60

CACHE_FILE = os.path.join(SAVE_DIR, ".last_index")
# ================================================

os.makedirs(SAVE_DIR, exist_ok=True)

_state = {"last_saved": -1, "dirty": False}


# ---------------- 缓存 ----------------

def write_cache(index: int) -> None:
    try:
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(index))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CACHE_FILE)
        _state["dirty"] = False
    except OSError as e:
        print(f"缓存写入失败: {e}")


def flush_cache() -> None:
    if _state["dirty"] and _state["last_saved"] >= 0:
        write_cache(_state["last_saved"])
        print(f"已保存缓存: {_state['last_saved']:06d}")


def read_cache() -> int:
    if not os.path.isfile(CACHE_FILE):
        return -1
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return int(content) if content else -1
    except (ValueError, OSError):
        return -1


def scan_last_index(save_dir: str) -> int:
    if not os.path.isdir(save_dir):
        return -1
    max_idx = -1
    pattern = re.compile(r"^(\d{6})(?:_\d+)?\.jpg$", re.IGNORECASE)
    for name in os.listdir(save_dir):
        m = pattern.match(name)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx


def get_start_index() -> int:
    cached = read_cache()
    if cached >= 0:
        print(f"从缓存读取到最后编号: {cached:06d}")
        return cached + 1
    scanned = scan_last_index(SAVE_DIR)
    if scanned >= 0:
        print(f"缓存不可用，扫描目录得到最后编号: {scanned:06d}")
        write_cache(scanned)
        return scanned + 1
    print("无历史记录，从 000000.jpg 开始")
    return 0


# ---------------- 信号处理 ----------------

def setup_signal_handlers():
    def handler(signum, frame):
        print(f"\n收到信号 {signum}，正在退出...")
        flush_cache()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, handler)


# ---------------- 全局 Geeked 实例 ----------------

_geeked: Geeked = None


def get_geeked() -> Geeked:
    global _geeked
    if _geeked is None:
        _geeked = Geeked(captcha_id=CAPTCHA_ID, captcha_type=CAPTCHA_TYPE)
    return _geeked


# ---------------- 采集 ----------------

# ==================== 配置 ====================
PROXIES_PER_ROUND = 5          # 每轮尝试的代理数
PROXY_VALIDATE = True          # 是否在使用前快速验证代理
PROXY_VALIDATE_TIMEOUT = 5     # 验证超时(秒)
# ================================================


def fetch_one(index: int) -> bool:
    """
    采集单张背景图。
    每轮尝试 PROXIES_PER_ROUND 个代理，全部失败才退避。
    """
    for attempt in range(MAX_RETRY):
        # ---- 内层：本轮尝试多个代理 ----
        for proxy_try in range(PROXIES_PER_ROUND):
            proxy = get_proxy()
            # if not proxy:
            #     print(f"[{index}] 代理池无可用代理，等待 1s")
            #     time.sleep(1)
            #     break  # 跳出内层，进入下一轮退避
            #
            # # 可选：快速验证代理可用性
            # if PROXY_VALIDATE and not validate_proxy(
            #     proxy, timeout=PROXY_VALIDATE_TIMEOUT
            # ):
            #     print(f"[{index}] 代理 {proxy} 验证失败，删除")
            #     delete_proxy(proxy)
            #     continue  # 换下一个代理，不退避

            try:
                g = get_geeked()
                g.set_proxy(proxy)

                data = g.load_captcha()

                # 字段名改为 captcha_type
                if data.get("captcha_type") != "icon":
                    print(f"[{index}] 跳过非 icon: {data.get('captcha_type')}")
                    return False

                img_paths = data.get("imgs")
                if not img_paths:
                    print(f"[{index}] imgs 字段缺失")
                    return False

                if isinstance(img_paths, str):
                    img_paths = [img_paths]

                saved_any = False
                for i, img_path in enumerate(img_paths):
                    img_url = f"{STATIC_BASE}/{img_path}"
                    resp = g.download(img_url, timeout=10)

                    if resp.status_code != 200:
                        delete_proxy(proxy)
                        print(f"[{index}] 下载失败 {resp.status_code}，"
                              f"标记代理失效")
                        saved_any = False
                        break  # 跳出图片循环，换下一个代理

                    suffix = f"_{i}" if len(img_paths) > 1 else ""
                    filepath = os.path.join(
                        SAVE_DIR, f"{index:06d}{suffix}.jpg"
                    )
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    print(f"[{index}] 已保存 {filepath} (代理 {proxy})")
                    saved_any = True

                if saved_any:
                    _state["last_saved"] = index
                    _state["dirty"] = True
                    write_cache(index)
                    time.sleep(random.uniform(*SLEEP_RANGE))
                    return True
                # 下载失败，继续试下一个代理
                continue

            except Exception as e:
                # 请求异常：删除代理，换下一个，不退避
                print(f"[{index}] 代理 {proxy} 出错: {e}")
                delete_proxy(proxy)
                continue

        # ---- 本轮所有代理都失败，退避后进入下一轮 ----
        backoff = min(
            BACKOFF_BASE ** attempt + random.uniform(0, 1),
            BACKOFF_MAX
        )
        print(f"[{index}] 本轮 {PROXIES_PER_ROUND} 个代理均失败，"
              f"退避 {backoff:.1f}s")
        time.sleep(backoff)

    print(f"[{index}] 重试 {MAX_RETRY} 轮后放弃")
    return False


def main():
    atexit.register(flush_cache)
    setup_signal_handlers()

    saved = get_start_index()
    fail_streak = 0

    while saved < TOTAL:
        try:
            ok = fetch_one(saved)
            if ok:
                saved += 1
                fail_streak = 0
            else:
                fail_streak += 1

            if fail_streak >= 3:
                cooldown = random.uniform(60, 120)
                print(f"连续失败 {fail_streak} 次，冷却 {cooldown:.0f}s")
                time.sleep(cooldown)
                fail_streak = 0

        except KeyboardInterrupt:
            print(f"\n用户中断，已保存到 {_state['last_saved']:06d}")
            break
        except Exception as e:
            print(f"主循环异常: {e}")
            time.sleep(10)

    print(f"采集完成，当前最大编号: {_state['last_saved']:06d}")


if __name__ == "__main__":
    main()