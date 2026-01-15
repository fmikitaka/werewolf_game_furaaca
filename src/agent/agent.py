"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []

        load_dotenv(Path(__file__).resolve().parents[2] / "config" / ".env", override=False)

    @staticmethod
    def _require_env_var(var_name: str) -> str:
        """Fetch an environment variable following provider recommendations."""
        value = os.getenv(var_name)
        if not value:
            msg = (
                f"{var_name} is not set. Set the provider API key as an environment variable "
                "before launching the agent (see the OpenAI / Google credential docs)."
            )
            raise RuntimeError(msg)
        return value

    @staticmethod
    def _normalize_request_name(request: Request | None) -> str | None:
        """Normalize request enum/object into lowercase string."""
        if request is None:
            return None
        if hasattr(request, "lower"):
            return request.lower()
        return str(request).lower()

    def _serialize_talks(self, history: list[Talk], limit: int, only_self: bool = False) -> list[dict[str, str]]:
        """Serialize talk history for prompt templates."""
        if limit <= 0:
            limit = len(history)
        serialized: list[dict[str, str]] = []
        for talk in history:
            agent_name = getattr(talk, "agent", "")
            if only_self and agent_name != self.agent_name:
                continue
            serialized.append(
                {
                    "agent": agent_name,
                    "text": getattr(talk, "text", ""),
                },
            )
        return serialized[-limit:]

    def _should_run_contradiction_check(self, request: Request | None) -> bool:
        """Determine whether contradiction check should run for the request."""
        config = self.config.get("contradiction_check", {})
        if not bool(config.get("enabled", False)):
            return False
        req_name = self._normalize_request_name(request)
        if req_name is None:
            return False
        targets = config.get("target_requests", [])
        normalized_targets = {str(target).lower() for target in targets}
        return req_name in normalized_targets

    @staticmethod
    def _parse_contradiction_output(raw_output: str) -> dict[str, Any]:
        """Extract JSON payload from contradiction checker output."""
        text = raw_output.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        payload = match.group(0) if match else text
        try:
            data: dict[str, Any] = json.loads(payload)
        except json.JSONDecodeError:
            data = {"is_contradiction": False, "explanation": "failed_to_parse"}
        data.setdefault("explanation", "")
        data.setdefault("is_contradiction", False)
        data.setdefault("revised_response", "")
        data["raw_output"] = raw_output
        return data

    def _run_contradiction_check(self, request: Request | None, candidate_response: str) -> dict[str, Any] | None:
        """Invoke contradiction check prompt and return parsed result."""
        if self.llm_model is None:
            return None
        template_str = (
            self.config.get("module_prompt", {})
            or {}
        ).get("contradiction_check")
        if not template_str:
            return None
        check_config = self.config.get("contradiction_check", {})
        history_limit = int(check_config.get("history_limit", 10))
        allow_role_bluff = False
        if self.role:
            role_value = str(getattr(self.role, "value", self.role))
            allowed_roles = {
                str(role_name).upper()
                for role_name in check_config.get("allow_role_bluff_roles", [])
            }
            allow_role_bluff = role_value.upper() in allowed_roles
        template = Template(template_str)
        rendered_prompt = template.render(
            agent_name=self.agent_name,
            role=self.role,
            info=self.info,
            request=self._normalize_request_name(request),
            agent_history=self._serialize_talks(self.talk_history, history_limit, only_self=True),
            recent_talks=self._serialize_talks(self.talk_history, history_limit),
            candidate_response=candidate_response,
            allow_role_bluff=allow_role_bluff,
        ).strip()
        try:
            raw_output = (self.llm_model | StrOutputParser()).invoke([HumanMessage(content=rendered_prompt)])
        except Exception:
            self.agent_logger.logger.exception("Failed to run contradiction check")
            return None
        return self._parse_contradiction_output(raw_output)

    def _postprocess_response(self, request: Request | None, prompt: str, response: str) -> str:
        """Apply optional post-processing such as normalization and contradiction checks."""
        normalized_request = self._normalize_request_name(request)
        response = response.strip()
        response = self._normalize_action_response(normalized_request, response)
        if not self._should_run_contradiction_check(request):
            return response
        result = self._run_contradiction_check(request, response)
        if not result:
            return response
        if result.get("is_contradiction"):
            explanation = result.get("explanation", "")
            self.agent_logger.logger.warning(
                "LLM response contradicted history: %s",
                explanation,
            )
            revised = result.get("revised_response") or result.get("suggested_response") or ""
            if revised and bool(self.config.get("contradiction_check", {}).get("apply_revision", False)):
                self.agent_logger.logger.info(
                    ["LLM_CONTRADICTION_REVISION", prompt, revised],
                )
                return str(revised).strip()
            if bool(self.config.get("contradiction_check", {}).get("fail_on_unresolved", False)):
                raise ValueError("Contradiction detected but no revision available")
        else:
            self.agent_logger.logger.debug(
                "Contradiction check passed for request: %s",
                request,
            )
        return response

    def _normalize_action_response(self, request_name: str | None, response: str) -> str:
        """Ensure action responses contain only agent names when required."""
        if request_name not in {"divine", "guard", "vote", "attack"}:
            return response
        alive_agents = self.get_alive_agents()
        if not alive_agents:
            return response
        candidate = response.strip()
        if not candidate:
            return candidate
        for agent in alive_agents:
            if candidate == agent:
                return agent
        for agent in alive_agents:
            if agent in candidate:
                return agent
        tokens = re.split(r"[\\s,:：、，。]+", candidate)
        for token in tokens:
            if token in alive_agents:
                return token
        return candidate

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        """Send message to LLM and get response.

        LLMにメッセージを送信して応答を取得する.

        Args:
            request (Request | None): The request type to process / 処理するリクエストタイプ

        Returns:
            str | None: LLM response or None if error occurred / LLMの応答またはエラー時はNone
        """
        if request is None:
            return None
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
        }
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            self.llm_message_history.append(HumanMessage(content=prompt))
            response = (self.llm_model | StrOutputParser()).invoke(self.llm_message_history)
            response = self._postprocess_response(request, prompt, response)
            self.llm_message_history.append(AIMessage(content=response))
            self.agent_logger.logger.info(["LLM", prompt, response])
            self.agent_logger.conversation(self.request, prompt, response)
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response

    @timeout
    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """
        if self.info is None:
            return

        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self._require_env_var("OPENAI_API_KEY")
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                )
            case "google":
                self._require_env_var("GOOGLE_API_KEY")
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self.llm_model = self.llm_model
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        response = self._send_message_to_llm(self.request)
        self.sent_talk_count = len(self.talk_history)
        return response or ""

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
