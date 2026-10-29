import streamlit as st
from groq import Groq
from tavily import TavilyClient

# --------------------------------------------------
# 1. 初期設定とAPIキーの読み込み
# --------------------------------------------------
st.set_page_config(page_title="リアルタイムWeb検索付き Groq AI", page_icon="🌐")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# --------------------------------------------------
# 2. 画面の初期化
# --------------------------------------------------
st.title("🌐 リアルタイムWeb検索付き Groq AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 3. チャットの処理（入力〜検索〜AI回答）
# --------------------------------------------------
if prompt := st.chat_input("質問を入力してください..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        search_context = ""
        
        # --- Web検索処理 ---
        with st.spinner("🔍 Webを検索中..."):
            try:
                search_result = tavily_client.search(query=prompt, search_depth="basic")
                search_context = "\n".join([result["content"] for result in search_result.get("results", [])])
            except Exception as e:
                st.warning("Web検索でエラーが発生しました。検索なしで回答します。")
                search_context = ""

        # --- Groq AI 呼び出し処理 ---
        with st.spinner("🤖 AIが考え中..."):
            
            # パンク対策①: 検索結果を1000文字に制限
            safe_search = str(search_context)[:1000] if search_context else ""
            
            final_user_prompt = "以下の[Web検索結果]を参照して、質問に日本語で分かりやすく回答してください。\n\n[Web検索結果]:\n" + safe_search + "\n\n[質問]:\n" + str(prompt)

            # パンク対策②: 過去の履歴を直近2件（1往復）に絞る
            safe_history = []
            recent_messages = st.session_state.messages[:-1][-2:]
            
            for m in recent_messages:
                if len(safe_history) == 0 and m["role"] != "user":
                    continue
                if len(safe_history) > 0 and safe_history[-1]["role"] == m["role"]:
                    continue
                
                safe_content = str(m["content"])
                if len(safe_content) > 200:
                    safe_content = safe_content[:200] + "..."
                    
                safe_history.append({"role": m["role"], "content": safe_content})
                
            if len(safe_history) > 0 and safe_history[-1]["role"] == "user":
                safe_history.pop()

            messages_to_send = safe_history + [{"role": "user", "content": final_user_prompt}]

            # パンク対策③: 音声用・検閲用(guard)・規約同意系モデルを除外してチャットモデルを自動選択
            available_models = [m.id for m in client.models.list().data]
            ng_words = ["whisper", "canopy", "orpheus", "guard", "tts", "playai", "vision"]
            text_models = [
                m for m in available_models 
                if not any(ng in m.lower() for ng in ng_words)
            ]
            
            candidate_models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "qwen-qwq-32b",
                "deepseek-r1-distill-llama-70b",
                "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
            
            selected_model = next((m for m in candidate_models if m in text_models), text_models[0] if text_models else available_models[0])
            
            with st.expander("ℹ️ 使用中のAIモデル情報"):
                st.write("選択されたモデル: " + str(selected_model))

            # パンク対策④: max_tokensを512に設定して上限エラーを完全回避
            try:
                completion = client.chat.completions.create(
                    model=selected_model,
                    messages=messages_to_send,
                    max_tokens=512,
                )
                
                full_response = completion.choices[0].message.content
                message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error("🚨 Groqエラー: " + str(e))
                st.stop()
