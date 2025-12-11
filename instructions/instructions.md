# AIWolf LLM エージェント（kanolab）プロジェクトメモ

Codex にこの内容を前提として作業してもらうためのコンテキスト資料。

## 1. プロジェクト概要

本プロジェクトでは、人狼知能プロジェクト（AI Wolf Project）の自然言語部門（LLM Division）で動作する LLM ベースの人狼エージェント（kanolab） を作成する。

目標：

役職別ロジックを LLM で自然かつ一貫性のある形で実現する

analysis → strategy → utterance の3段階に分けて、より論理的な発話行動を生成する

ゲームログを後で学習データとして扱い、戦略改善に活かす

エージェントは、

initialize

daily_initialize

analyze

decide_strategy

talk_from_strategy

vote

などのフェーズに従い、ゲームサーバと通信する。

## 2. エージェント設計思想（アーキテクチャ）
### 2.1 コンポーネント構成
AIWolf Server
   ↓↑
NL Bridge（ゲーム状態 → プロンプト入力形式）
   ↓↑
LLM Agent Core
   ├─ Analysis Module
   ├─ Strategy Module
   └─ Utterance Module

### 2.2 各モジュールの役割

Analysis

発言ログ・イベント情報（CO/霊能/占い結果など）をもとに、
各プレイヤーの怪しさ・信頼度を JSON で推定する。

Strategy

今日吊りたい候補

今日守りたい候補

発言方針（論理的・強気・弱気など）

投票先の第一・第二候補
を決める。

Utterance

Strategy に従って 1ターン分の発言を生成。

過去発言との一貫性を維持する。

## 3. 共通プロンプト設計

ここでは Codex が参照すべき、最新のプロンプトテンプレートを記録する。

### 3.1 initialize
prompt:
  initialize: |-
    あなたは人狼ゲームのエージェントです。
    あなたの名前は {{ info.agent }} です。
    あなたの役職は {{ role.value }} です。

    これから人狼ゲームを開始します。
    まず簡単な自己紹介と初日の方針を述べてください。

    出力形式：
    ```json
    {
      "self_introduction": "",
      "opening_policy": "",
      "message_to_others": ""
    }
    ```

### 3.2 daily_initialize
  daily_initialize: |-
    あなたは {{ info.agent }}、役職 {{ role.value }} です。

    {{ info.day }} 日目が始まりました。
    生存プレイヤー: {{ info.alive_agents }}
    直前の出来事:
    {{ info.latest_events }}

    今日の行動を考える準備として、状況を整理してください。

    出力形式：
    ```json
    {
      "situation_summary": "",
      "today_objective": "",
      "notes_for_myself": ""
    }
    ```

### 3.3 analyze（状況分析）
  analyze: |-
    === ゲーム情報 ===
    日数: {{ info.day }}
    生存: {{ info.alive_agents }}
    イベント:
    {{ info.structured_events }}

    === 発言（要約） ===
    {{ info.talk_summaries }}

    以下を JSON で出力：

    ```json
    {
      "suspicion_scores": {},
      "trust_scores": {},
      "player_explanations": {},
      "today_main_lynch_target": {
        "agent": "",
        "reason": ""
      },
      "today_never_lynch_candidates": []
    }
    ```

### 3.4 decide_strategy（戦略決定）
  decide_strategy: |-
    以下は直前の analysis 結果です：
    {{ info.last_analysis_json }}

    今日1日の戦略を JSON で出力してください：

    ```json
    {
      "talk_style": "",
      "main_attack_target": {
        "agent": "",
        "reason": ""
      },
      "sub_attack_targets": [],
      "protect_targets": [],
      "vote_plan": {
        "primary": "",
        "backup": ""
      },
      "today_talking_points": []
    }
    ```

### 3.5 talk_from_strategy（実際の発言生成）
  talk_from_strategy: |-
    戦略：
    {{ info.today_strategy_json }}

    この戦略に従い、1ターン分の発言だけ生成してください。
    日本語の文章のみ出力。1～2行が望ましい。

### 3.6 vote（投票決定）
  vote: |-
    戦略：
    {{ info.today_strategy_json }}

    今日の投票先を次の JSON 形式で返してください：

    ```json
    {
      "vote_target": "",
      "reason": ""
    }
    ```

## 4. ロール別追加方針メモ
### 4.1 村人／市民 (VILLAGER)

主な情報源は他人の発言

占い師・霊能者の整合性を強く評価

投票理由は具体的に

特に「話題そらし」「急な発言転換」を重要視する

### 4.2 占い師 (SEER)

占い結果を必ず整合的に扱う

黒を見つけた場合は基本的に公表

真狼狂のライン判断を逐次更新

狂人の騙りパターンを推定する

### 4.3 人狼 (WEREWOLF)

潜伏 or 騙りを事前に方針として持つ

相方とのライン管理（つなぐ／切る）

村同士の対立を煽る

「他者に矛先を移す理由」を自然に用意する

### 4.4 狂人 (POSSESSED)

占い師の真偽を混乱させる

人狼に都合が良いライン形成

村人の論理の邪魔をする

黒出し先や投票先を柔軟に変える

## 5. 実装側の呼び出しフロー（Python想定）
# 1日目開始
daily_info = call_llm("daily_initialize", state)

analysis_json = call_llm("analyze", state)
state["analysis"] = json.loads(analysis_json)

strategy_json = call_llm("decide_strategy", {
    **state,
    "last_analysis_json": analysis_json
})
state["today_strategy"] = json.loads(strategy_json)

# talk
utterance = call_llm("talk_from_strategy", {
    **state,
    "today_strategy_json": strategy_json
})

# vote
vote_json = call_llm("vote", {
    **state,
    "today_strategy_json": strategy_json
})
vote = json.loads(vote_json)["vote_target"]

## 6. Codex への指示例

このファイルを基に Codex へ指示するときの例：

codex work "docs/aiwolf-notes/2025-12-11-kanolab-codex-context.md を参照し、kanolab の LLM 戦略モジュールを実装してください"


または Slack では：

@Codex このメモを読み込んで、占い師用の decide_strategy プロンプトを改善してください

## 7. 今後追加すべき項目（TODO）

過去ログから特徴量抽出する Python スクリプト

meta-strategy（ゲーム全体の方針）

LLM の「矛盾チェック」用サブルーチン

投票履歴の一貫性管理

ロール別テストシナリオ集

## End of File