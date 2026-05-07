#!/usr/bin/env python3
"""Register the RoboMME config name in wan_va/configs/__init__.py.

Run once from the lingbot-va repository root after copying
va_robomme_cfg.py and va_robomme_train_cfg.py into wan_va/configs/.
"""

from __future__ import annotations

from pathlib import Path


INIT_PATH = Path("wan_va/configs/__init__.py")
IMPORT_LINES = [
    "from .va_robomme_cfg import va_robomme_cfg",
    "from .va_robomme_train_cfg import va_robomme_train_cfg",
]
DICT_LINES = [
    "    'robomme': va_robomme_cfg,",
    "    'robomme_train': va_robomme_train_cfg,",
]


def main() -> None:
    text = INIT_PATH.read_text(encoding="utf-8")

    for line in IMPORT_LINES:
        if line not in text:
            # Insert after the last existing config import.
            lines = text.splitlines()
            insert_at = 0
            for i, current in enumerate(lines):
                if current.startswith("from .va_"):
                    insert_at = i + 1
            lines.insert(insert_at, line)
            text = "\n".join(lines) + "\n"

    for line in DICT_LINES:
        if line not in text:
            marker = "}"
            pos = text.rfind(marker)
            if pos == -1:
                raise RuntimeError("Could not find VA_CONFIGS closing brace")
            text = text[:pos] + line + "\n" + text[pos:]

    INIT_PATH.write_text(text, encoding="utf-8")
    print(f"Registered robomme configs in {INIT_PATH}")


if __name__ == "__main__":
    main()
