# eval_pass_rate.py
"""
连续跑 N 次极验验证，统计通过率和失败原因。
"""
import os
import sys

# 切换到项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import time
from geeked.geeked import Geeked

# ==================== 配置 ====================
CAPTCHA_ID = "af29b3003fc94f2ba29e865b31ee86ee"
CAPTCHA_TYPE = "icon"
N = 20                       # 测试次数
SLEEP_BETWEEN = 3            # 每次间隔(秒)
RETRY_ON_FAIL = False        # 失败是否重试
# ================================================


def run_once(idx):
    """跑一次验证，返回 (成功与否, 失败原因)"""
    try:
        g = Geeked(captcha_id=CAPTCHA_ID, captcha_type=CAPTCHA_TYPE)
        result = g.solve()

        # solve() 成功时返回 dict，包含 pass_token
        if result and result.get("pass_token"):
            return True, None
        return False, "no_pass_token"

    except Exception as e:
        msg = str(e)

        # 分析失败原因
        if "Failed to submit captcha" in msg:
            if "'result': 'fail'" in msg:
                return False, "verify_fail"
            return False, "submit_error"
        elif "load_captcha" in msg or "format_response" in msg:
            return False, "load_error"
        else:
            return False, f"exception: {msg[:80]}"


def main():
    success = 0
    fail_reasons = {}

    print(f"开始评估: {N} 次")
    print(f"captcha_id: {CAPTCHA_ID}")
    print(f"captcha_type: {CAPTCHA_TYPE}")
    print("=" * 60)

    for i in range(1, N + 1):
        start = time.time()
        ok, reason = run_once(i)
        elapsed = time.time() - start

        if ok:
            success += 1
            print(f"[{i:2d}/{N}] ✅ 成功  ({elapsed:.2f}s)")
        else:
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            print(f"[{i:2d}/{N}] ❌ 失败  ({elapsed:.2f}s)  原因: {reason}")

        if i < N:
            time.sleep(SLEEP_BETWEEN)

    # 统计
    print("=" * 60)
    print(f"总次数: {N}")
    print(f"成功: {success}")
    print(f"失败: {N - success}")
    print(f"通过率: {success / N * 100:.1f}%")

    if fail_reasons:
        print("\n失败原因分布:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count} 次")

    # 保存结果
    with open("eval_result.txt", "w", encoding="utf-8") as f:
        f.write(f"总次数: {N}\n")
        f.write(f"成功: {success}\n")
        f.write(f"通过率: {success / N * 100:.1f}%\n")
        for reason, count in fail_reasons.items():
            f.write(f"{reason}: {count}\n")


if __name__ == "__main__":
    main()