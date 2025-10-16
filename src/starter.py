"""Module for launching agents.

エージェントを起動するためのモジュール.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from utils.agent_utils import init_agent_from_packet

if TYPE_CHECKING:
    from agent.agent import Agent

from time import sleep

from aiwolf_nlp_common.client import Client
from aiwolf_nlp_common.packet import Request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_client(config: dict[str, Any]) -> Client:
    """Create a client.

    クライアントの作成.

    Args:
        config (dict[str, Any]): Configuration dictionary / 設定辞書

    Returns:
        Client: Created client instance / 作成されたクライアントインスタンス
    """
    return Client(
        url=str(config["web_socket"]["url"]),
        token=(str(config["web_socket"]["token"]) if config["web_socket"]["token"] else None),
    )


def connect_to_server(client: Client, name: str) -> None:
    """Handle connection to the server.

    サーバーへの接続処理.

    Args:
        client (Client): Client instance / クライアントインスタンス
        name (str): Agent name / エージェント名
    """
    while True:
        try:
            client.connect()
            logger.info("エージェント %s がゲームサーバに接続しました", name)
            break
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "エージェント %s がゲームサーバに接続できませんでした",
                name,
            )
            logger.warning(ex)
            logger.info("再接続を試みます")
            sleep(15)


def handle_game_session(
    client: Client,
    config: dict[str, Any],
    name: str,
) -> None:
    """Handle game session.

    ゲームセッションの処理.

    Args:
        client (Client): Client instance / クライアントインスタンス
        config (dict[str, Any]): Configuration dictionary / 設定辞書
        name (str): Agent name / エージェント名
    """
    agent: Agent | None = None
    while True:
        packet = client.receive()
        if packet.request == Request.NAME:
            client.send(name)
            continue
        if packet.request == Request.INITIALIZE:
            agent = init_agent_from_packet(config, name, packet)
        if not agent:
            raise ValueError(agent, "エージェントが初期化されていません")
        agent.set_packet(packet)
        req = agent.action()
        agent.agent_logger.packet(agent.request, req)
        if req:
            client.send(req)
        if packet.request == Request.FINISH:
            break


def connect(config: dict[str, Any], idx: int = 1) -> None:
    """Launch an agent.

    エージェントを起動する.

    Args:
        config (dict[str, Any]): Configuration dictionary / 設定辞書
        idx (int): Agent index (default: 1) / エージェントインデックス (デフォルト: 1)
    """
    name = str(config["agent"]["team"]) + str(idx)
    while True:
        try:
            client = create_client(config)
            connect_to_server(client, name)
            try:
                handle_game_session(client, config, name)
            finally:
                client.close()
                logger.info("エージェント %s とゲームサーバの接続を切断しました", name)
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "エージェント %s がエラーで終了しました",
                name,
            )
            logger.warning(ex)

        if not bool(config["web_socket"]["auto_reconnect"]):
            break
