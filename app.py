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

     # 検索結果と指示をユーザーの質問文と合体させる
        final_user_prompt = f"""以下の【Web検索結果】を参照して、質問に日本語で分かりやすく回答してください。

【Web検索結果】:
{search_context}

【質問】:
{prompt}"""

        # 過去履歴を取得
        messages_to_send = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        messages_to_send.append({"role": "user", "content": final_user_prompt})

        # --- 現在利用可能なモデルをGroqから自動取得 ---
        available_models = [m.id for m in client.models.list().data]
        
        # 優先順位リスト（利用可能なものから最初に見つかったものを採用）
        candidate_models = [
            "llama-3.3-70b-versatile",
            "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]
        
        # 候補に合致するものを探す（無ければ現在動いている最初のチャットモデルを採用）
        selected_model = next((m for m in candidate_models if m in available_models), available_models[0])
        
        # どのモデルが選ばれたかを画面の折りたたみメニューで確認できるように表示
        with st.expander("ℹ️ 使用中のAIモデル情報"):
            st.write(f"選択されたモデル: `{selected_model}`")
            st.write("現在利用可能なモデル一覧:", available_models)

        # Groq API呼び出し
        completion = client.chat.completions.create(
            model=selected_model,
            messages=messages_to_send,
        )

        response_text = completion.choices[0].message.content
        st.markdown(response_text)

    # 返答を履歴に追加
    st.session_state.messages.append({"role": "assistant", "content": response_text})
