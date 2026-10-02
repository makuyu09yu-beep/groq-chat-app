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
 # 🚨 文字数オーバー対策①: 今回の検索結果を1500文字で安全にカット
        safe_search = str(search_context)[:1500] if search_context else ""
        
        final_user_prompt = "以下の[Web検索結果]を参照して、質問に日本語で分かりやすく回答してください。\n\n[Web検索結果]:\n" + safe_search + "\n\n[質問]:\n" + str(prompt)

        # 🚨 文字数オーバー対策②: 過去の履歴が雪だるま式に増えるのを防ぐ最強フィルター
        safe_history = []
        # 全履歴を送るのをやめ、直近4件（2往復分）だけを取り出す
        recent_messages = st.session_state.messages[:-1][-4:]
        
        for m in recent_messages:
            if len(safe_history) == 0 and m["role"] != "user":
                continue
            if len(safe_history) > 0 and safe_history[-1]["role"] == m["role"]:
                continue
                
            # 過去の会話に混ざっている古い検索結果を500文字でバッサリ切る（記憶容量の節約）
            safe_content = str(m["content"])
            if len(safe_content) > 500:
                safe_content = safe_content[:500] + "\n...(以前の検索結果は省略)..."
                
            safe_history.append({"role": m["role"], "content": safe_content})
            
        if len(safe_history) > 0 and safe_history[-1]["role"] == "user":
            safe_history.pop()

        messages_to_send = safe_history + [{"role": "user", "content": final_user_prompt}]

        available_models = [m.id for m in client.models.list().data]
        text_models = [m for m in available_models if "whisper" not in m.lower()]
        
        candidate_models = [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]
        
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
