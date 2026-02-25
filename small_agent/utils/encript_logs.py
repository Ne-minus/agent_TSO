import base64
from typing import Union

# Можно захардкодить, а лучше прокинуть через ENV (но ты просила без файлов — ок)
DEFAULT_KEY = "my-not-secret-key"


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def encrypt_obfuscate(text: str, key: str = DEFAULT_KEY) -> str:
    """
    Лёгкая обратимая обфускация для логов.
    Возвращает URL-safe base64 строку.
    """
    if text is None:
        return ""
    data = text.encode("utf-8")
    k = key.encode("utf-8")
    x = _xor_bytes(data, k)
    return base64.urlsafe_b64encode(x).decode("ascii")


def decrypt_obfuscate(token: str, key: str = DEFAULT_KEY) -> str:
    """
    Обратное преобразование.
    """
    if not token:
        return ""
    x = base64.urlsafe_b64decode(token.encode("ascii"))
    k = key.encode("utf-8")
    data = _xor_bytes(x, k)
    return data.decode("utf-8")


def encript_ticketdata(payload_log: dict) -> dict:
    for i in payload_log:
        if i in ["callerId", "initiatorId"]:
            payload_log[i] = encrypt_obfuscate(payload_log[i])
        elif i == "information":
            for j in payload_log[i]:
                if j["label"] in ["Телефон для связи", "ФИО ВК", "Помещение"]:
                    j["value"] = encrypt_obfuscate(j["value"])
                elif j["label"] == "Адрес":
                    addr = j["value"].split(", ")
                    addr[-1] = encrypt_obfuscate(addr[-1])
                    addr = ", ".join(addr)
                    j["value"] = addr
    return payload_log
