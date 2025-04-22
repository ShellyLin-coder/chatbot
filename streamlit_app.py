import streamlit as st
import google.generativeai as genai
from datetime import datetime
import csv
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import altair as alt
import numpy as np

# ---------------- Page Menu ----------------
MENU_OPTIONS = {
    "Chatbot": "💬 Chatbot",
    "Dashboard": "📊 Dashboard",
}

with st.sidebar:
    st.image("UIC_BUSINESS_LOGO.PNG", width=200)
    st.image("Soultalk logo.png", width=80)
    st.write("Detecting Suicidal Ideation and Depression in Real-Time Using a Conversational AI Trained on Reddit Dataset")
    st.markdown("## 📚 Navigation")
    for page_id, page_label in MENU_OPTIONS.items():
        if st.button(page_label):
            st.session_state.page = page_id
            if page_id != "Dashboard":
                st.session_state.authenticated = False  # Reset login status when leaving Dashboard

    st.markdown("---")  # Divider
    gemini_api_key = st.text_input("🔐 Gemini API Key", type="password", key="gemini_api")
    st.markdown("[Get Gemini API Key](https://makersuite.google.com/app/apikey)")




# Initialize page status
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

selected_page = st.session_state.page

# ---------------- Chatbot page ----------------
if selected_page == "Chatbot":
    
    # Show Disclaimer popup (will only be shown once)
    if "seen_disclaimer" not in st.session_state:
        st.session_state.seen_disclaimer = False

    if not st.session_state.seen_disclaimer:
        st.warning("⚠️ **Disclaimer**: Your input may be stored and analyzed for research or improvement purposes.")
        if st.button("✅ I Understand"):
            st.session_state.seen_disclaimer = True
            st.rerun()
        st.stop()

    # Clear conversation button
    if st.sidebar.button("🔄 Clear Chat"):
        st.session_state.chat_history = [
            ("assistant", "Hello, I'm here for you. How are you feeling today?")
        ]

    # UI title
    st.title("🌼 Mental Wellness Buddy")
    st.caption('💬 Using "Google Gemini API" to provide sentiment support')

    # Initial conversation record
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("assistant", "Hello, I'm here for you. How are you feeling today?")
        ]

    # Show conversation history
    for role, content in st.session_state.chat_history:
        st.chat_message(role).write(content)

    # User input processing
    if prompt := st.chat_input("Please enter your thoughts or feelings...", key="chat_input_main"):
        if not gemini_api_key:
            st.warning("⚠️ Please enter your Gemini API key.")
            st.stop()

        # Recording user input
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").write(prompt)

        # Save input to csv
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("user_input_log.csv", "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([timestamp, prompt])

        # Call Gemini API response
        try:
            genai.configure(api_key=gemini_api_key)

            # Freely switch which model to use here
            model_name = "gemini-2.0-flash"  # Or change to "gemini-pro"
            model = genai.GenerativeModel(model_name)

            # Create a chat object
            chat = model.start_chat(history=[])

            # Add psychological care system prompt (auto processed according to the model)
            system_prompt = (
                "You are a kind and empathetic mental health support assistant. "
                "Respond gently, and give short encouraging advice when needed."
            )

            if "pro" in model_name:
                # Only use it if it supports system_instruction
                chat = model.start_chat(
                    history=[],
                    system_instruction=system_prompt
                )
            else:
                # Flash or other models are not supported ➜ Send in this way
                chat.send_message(system_prompt)

            # Send user message
            response = chat.send_message(prompt)
            reply = response.text

            # Save Response
            st.session_state.chat_history.append(("assistant", reply))
            st.chat_message("assistant").write(reply)

        except Exception as e:
            st.error(f"❌ Error occurred: {e}")


# ---------------- Dashboard page ----------------
elif selected_page == "Dashboard":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if username == "localhost" and password == "Demo1234":
                    st.session_state.authenticated = True
                    st.success("✅ Login successful!")
                else:
                    st.error("❌ Invalid credentials")
    else:
        st.title("📊 Chatbot Dashboard")
        st.success("🔓 Logged in")

        if st.button("🚪 Log out"):
            st.session_state.authenticated = False
            st.rerun()

        try:
            df = pd.read_csv("user_input_log.csv", names=["timestamp", "prompt"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

            tab1, tab2, tab3 = st.tabs(["🗂 Table", "📈 Stats", "☁️ Word Cloud"])

            with tab1:
                st.subheader("📋 User Responses")
                st.dataframe(df)

            with tab2:
                # Chart 1
                view_option = st.radio("Select Time Granularity:", ["Minute", "Hour", "Date"], horizontal=True)

                if view_option == "Minute":
                    df_grouped = df.groupby(df["timestamp"].dt.floor("min"))["prompt"].count().reset_index()
                elif view_option == "Hour":
                    df_grouped = df.groupby(df["timestamp"].dt.floor("h"))["prompt"].count().reset_index()
                else:  # Date
                    df_grouped = df.groupby(df["timestamp"].dt.date)["prompt"].count().reset_index()

                if "timestamp" in df_grouped.columns:
                    df_grouped.rename(columns={"prompt": "Messages", "timestamp": "time"}, inplace=True)
                elif df_grouped.columns[0] != "time":
                    df_grouped.rename(columns={df_grouped.columns[0]: "time", "prompt": "Messages"}, inplace=True)

                st.subheader(f"📊 Message Count Over Time (by {view_option})")
                line = alt.Chart(df_grouped).mark_line(point=True).encode(
                    x=alt.X("time:T", title=view_option),
                    y=alt.Y("Messages:Q", title="Message Count", scale=alt.Scale(nice=True), axis=alt.Axis(tickMinStep=1))
                ).properties(height=300)
                st.altair_chart(line, use_container_width=True)

                # Chart 2
                st.subheader("📏 Prompt Length Distribution")
                df["length"] = df["prompt"].astype(str).apply(len)
                df["length_bin"] = pd.cut(df["length"], bins=np.arange(0, df["length"].max() + 5, 5), right=False)
                df["length_bin_label"] = df["length_bin"].apply(lambda x: f"{int(x.left)+1}–{int(x.right)}")

                hist_data = df["length_bin_label"].value_counts().sort_index().reset_index()
                hist_data.columns = ["Prompt Length", "Count"]

                chart = alt.Chart(hist_data).mark_bar().encode(
                    x=alt.X("Prompt Length:N", title="Prompt Length", sort=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Count:Q", title="Count", axis=alt.Axis(tickMinStep=1))
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

                # Chart 3
                st.subheader("🔤 Most Common Words")
                words = " ".join(df["prompt"].astype(str).tolist()).split()
                word_freq = Counter(words)
                common_words = pd.DataFrame(word_freq.most_common(10), columns=["word", "count"])
                bar = alt.Chart(common_words).mark_bar().encode(
                    y=alt.Y("word", sort="-x"),
                    x="count"
                )
                st.altair_chart(bar, use_container_width=True)

            with tab3:
                st.subheader("☁️ Word Cloud of Prompts")
                text = " ".join(df["prompt"].astype(str).tolist())
                if text:
                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("No text data available for word cloud.")

        except FileNotFoundError:
            st.warning("⚠️ No input log found yet.")


# ---------------- Notice for Developers ----------------
# Dashboard - Username: localhost / Password: Demo1234 (Click login twice)

