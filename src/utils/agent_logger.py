"""Module defining agent logging functionality.

エージェントのログを出力するクラスを定義するモジュール.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ulid import ULID

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Request


class AgentLogger:
    """A class for handling agent logging.

    エージェントのログを出力するクラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
    ) -> None:
        """Initialize the agent logger.

        エージェントのログを初期化する.

        Args:
            config (dict[str, Any]): Configuration dictionary containing logging settings / ログ設定を含む設定辞書
            name (str): Name of the agent for logging / ログ用のエージェント名
            game_id (str): Game ID for log file organization / ログファイル整理用のゲームID
        """
        self.config = config
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(
            logging.getLevelNamesMapping()[str(self.config["log"]["level"]).upper()],
        )
        self.output_dir: Path | None = None
        self.conversation_log_path: Path | None = None
        self.conversation_all_path: Path | None = None
        if bool(self.config["log"]["console_output"]):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if bool(self.config["log"]["file_output"]):
            ulid: ULID = ULID.from_str(game_id)
            tz = datetime.now(UTC).astimezone().tzinfo
            self.output_dir = (
                Path(
                    str(self.config["log"]["output_dir"]),
                )
                / datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime(
                    "%Y%m%d%H%M%S%f",
                )[:-3]
            )
            self.output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            handler = logging.FileHandler(
                self.output_dir / f"{self.name}.log",
                mode="w",
                encoding="utf-8",
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            if bool(self.config["log"].get("conversation_output", False)):
                self.conversation_log_path = self.output_dir / f"{self.name}_conversation.txt"
                self.conversation_log_path.write_text("", encoding="utf-8")
            if bool(self.config["log"].get("conversation_all_output", False)):
                filename = str(
                    self.config["log"].get("conversation_all_filename", "conversation_all.txt"),
                )
                self.conversation_all_path = self.output_dir / filename
                if not self.conversation_all_path.exists():
                    self.conversation_all_path.write_text("", encoding="utf-8")

    def packet(self, req: Request | None, res: str | None) -> None:
        """Log packet information.

        パケットのログを出力.

        Args:
            req (Request | None): Request packet to log / ログ出力するリクエストパケット
            res (str | None): Response string to log / ログ出力するレスポンス文字列
        """
        if not req:
            return
        if req.lower() not in self.config["log"]["request"]:
            return
        if not bool(self.config["log"]["request"][req.lower()]):
            return
        if not res:
            self.logger.info([str(req)])
        else:
            self.logger.info([str(req), res])

    def conversation(self, req: Request | None, prompt: str, response: str) -> None:
        """Write LLM prompt/response conversation to log files."""
        timestamp = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
        request_name = str(req) if req else "None"
        entry = (
            f"[{timestamp}] {request_name} ({self.name})\n"
            "PROMPT:\n"
            f"{prompt}\n\n"
            "RESPONSE:\n"
            f"{response}\n\n"
            "---\n"
        )
        if self.conversation_log_path:
            with self.conversation_log_path.open("a", encoding="utf-8") as file:
                file.write(entry)
        if self.conversation_all_path:
            with self.conversation_all_path.open("a", encoding="utf-8") as file:
                file.write(entry)
