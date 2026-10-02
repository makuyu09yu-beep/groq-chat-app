import streamlit as st
from groq import Groq
from tavily import TavilyClient

# --------------------------------------------------
# 1. APIキーの設定（ご自身のキーに書き換えてください）
# --------------------------------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# --------------------------------------------------
# 2. 画面の初期化
# --------------------------------------------------
st.title("Groq AI WEB対応版")

# チャット履歴の保持
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去ログの表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 3. ユーザー入力と処理
# --------------------------------------------------
if prompt := st.chat_input("最新の価格やニュースなど、何でも質問してください"):
    # ユーザーの発言を表示
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AIの返答処理
    with st.chat_message("assistant"):
        # 処理中のステータス表示
        with st.status("🔍 情報を検索中...", expanded=True) as status:
            try:
                # Web検索を実行（上位3件の最新結果を取得）
                search_response = tavily.search(query=prompt, max_results=3)
                
                # 検索結果をテキストとして抽出
                search_context = "\n\n".join([
                    f"【タイトル】: {result['title']}\n【URL】: {result['url']}\n【内容】: {result['content']}"
                    for result in search_response.get("results", [])
                ])
                status.update(label="✅ 検索完了！AIが回答を生成中...", state="complete")
            except Exception as e:
                search_context = "検索中にエラーが発生しました。"
                status.update(label="⚠️ 検索スキップ", state="error")

        # システムプロンプト（検索結果を元に回答させる指示）
        system_instruction = f"""
あなたは最新情報を把握している優秀なリサーチアシスタントです。
以下の【Web検索結果】を参照して、ユーザーの質問に正確かつ分かりやすく回答してください。

【Web検索結果】:
{search_context}
"""

    # 検索結果と指示をユーザーの質問文と合体させる（エラー防止の安全な書き方）
# 🚨 パンク対策①: 検索結果をさらに安全な1000文字に制限
        safe_search = str(search_context)[:1000] if search_context else ""
        
        final_user_prompt = "以下の[Web検索結果]を参照して、質問に日本語で分かりやすく回答してください。\n\n[Web検索結果]:\n" + safe_search + "\n\n[質問]:\n" + str(prompt)

        # 🚨 パンク対策②: 過去の履歴を直近2件（1往復）に極限まで絞る
        safe_history = []
        recent_messages = st.session_state.messages[:-1][-2:]
        
        for m in recent_messages:
            if len(safe_history) == 0 and m["role"] != "user":
                continue
            if len(safe_history) > 0 and safe_history[-1]["role"] == m["role"]:
                continue
            
            # 過去の会話データも思い切って200文字でカット
            safe_content = str(m["content"])
            if len(safe_content) > 200:
                safe_content = safe_content[:200] + "..."
                
            safe_history.append({"role": m["role"], "content": safe_content})
            
        if len(safe_history) > 0 and safe_history[-1]["role"] == "user":
            safe_history.pop()

        messages_to_send = safe_history + [{"role": "user", "content": final_user_prompt}]

        available_models = [m.id for m in client.models.list().data]
        # whisper（音声）とcanopylabs（規約同意が必要な特殊モデル）を候補から除外
        text_models = [m for m in available_models if "whisper" not in m.lower() and "canopy" not in m.lower()]
        
        candidate_models = [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]
        
        selected_model = next((m for m in candidate_models if m in text_models), text_models[0] if text_models else available_models[0])
        
        with st.expander("ℹ️ 使用中のAIモデル情報"):
            st.write("選択されたモデル: " + str(selected_model))

        try:
            # 🚨 解決の決め手: AIの回答用メモリを1024トークンに制限して、自滅パンクを物理的に防ぐ
            completion = client.chat.completions.create(
                model=selected_model,
                messages=messages_to_send,
                max_tokens=1024,
            )
        except Exception as e:
            st.error("🚨 Groqエラー: " + str(e))
            st.stop()
        
        selected_model = next((m for m in candidate_models if m in text_models), text_models[0] if text_models else available_models[0])
        
        with st.expander("ℹ️ 使用中のAIモデル情報"):
            st.write(f"選択されたモデル: `{selected_model}`")

        try:
            completion = client.chat.completions.create(
                model=selected_model,
                messages=messages_to_send,
            )
        except Exception as e:
            st.error(f"🚨 Groqエラー: {e}")
            st.stop()
        
        response_text = completion.choices[0].message.content
        st.markdown(response_text)

    # 返答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response_text})
