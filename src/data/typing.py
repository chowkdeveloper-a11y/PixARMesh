from enum import IntEnum


class TokenType(IntEnum):
    COND_PREFIX = 0
    PADDING = 1
    SPECIAL_TOKEN = 2
    LAYOUT = 3
    OBJECT = 4
