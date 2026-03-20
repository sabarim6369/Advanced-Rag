import re

BLOCKED_QUERY_PATTERNS = [
    r"ignore instructions",
    r"\bgive salary\b",
    r"\bleak\b",
    r"\bapi[- ]?key\b",
    r"\bsecret\b",
    r"\btoken\b",
    r"\bpassword\b",
    r"\bcredential",
    r"\bprivate key\b",
    r"\baccess key\b",
    r"\bconnection string\b",
    r"\bclient secret\b",
    r"\benv\b",
    r"\.env\b",
]

SENSITIVE_LINE_PATTERNS = [
    r"api[_ -]?key",
    r"secret",
    r"token",
    r"password",
    r"passwd",
    r"credential",
    r"private[_ -]?key",
    r"access[_ -]?key",
    r"client[_ -]?secret",
    r"authorization:",
    r"bearer\s+[a-z0-9._\-]+",
    r"sk-[a-z0-9]+",
    r"ghp_[a-z0-9]+",
    r"AIza[0-9A-Za-z\-_]+",
    r"xox[baprs]-[A-Za-z0-9-]+",
]

BLOCK_MESSAGE = "Blocked due to security policy: sensitive data cannot be disclosed."


def check_query(query):
    lowered_query = query.lower()
    return not any(re.search(pattern, lowered_query) for pattern in BLOCKED_QUERY_PATTERNS)


def sanitize_context(text):
    safe_lines = []
    for line in text.splitlines():
        if _looks_sensitive(line):
            continue
        safe_lines.append(line)
    return "\n".join(safe_lines).strip()


def response_has_sensitive_data(text):
    return _looks_sensitive(text)


def safe_response_or_block(text):
    if response_has_sensitive_data(text):
        return BLOCK_MESSAGE
    return text


def _looks_sensitive(text):
    lowered_text = text.lower()
    for pattern in SENSITIVE_LINE_PATTERNS:
        if re.search(pattern, lowered_text, re.IGNORECASE):
            return True

    # Catch assignments that look like secrets without relying only on keywords.
    if re.search(r"[A-Za-z0-9_]{2,}\s*[:=]\s*['\"]?[A-Za-z0-9_\-\/+=]{16,}", text):
        return True

    return False
