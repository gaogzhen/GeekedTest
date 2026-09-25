# main.py
from utils import setup_root_logger
from geeked.geeked import Geeked


def main():
    setup_root_logger()

    captcha_id = "af29b3003fc94f2ba29e865b31ee86ee"
    g = Geeked(captcha_id=captcha_id, captcha_type="icon")
    result = g.solve()
    print(result)


if __name__ == "__main__":
    main()