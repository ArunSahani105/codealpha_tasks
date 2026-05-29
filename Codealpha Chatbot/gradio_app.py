"""
Gradio chatbot UI for the FAQ Chatbot.

Run:
    python -m app.gradio_app
or inside Google Colab:
    !python app/gradio_app.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple

import gradio as gr

# Allow running this file directly: `python app/gradio_app.py`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.chatbot_engine import FAQChatbot  # noqa: E402

DATASET_PATH = os.path.join(ROOT, "dataset", "faq.csv")
LOG_PATH = os.path.join(ROOT, "logs", "chat_logs.csv")

bot = FAQChatbot(dataset_path=DATASET_PATH, threshold=0.30, log_path=LOG_PATH)


CUSTOM_CSS = """
#chatbot { height: 520px; border-radius: 16px; }
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
.message.user { background: #2563eb !important; color: white !important;
    border-radius: 18px 18px 4px 18px !important; }
.message.bot { background: #1f2937 !important; color: #f3f4f6 !important;
    border-radius: 18px 18px 18px 4px !important; }
footer { display: none !important; }
"""


def respond(message: str, history: List[Tuple[str, str]]):
    """Generator that streams a typing-animation response."""
    history = history or []
    if not message or not message.strip():
        history.append((message or "", "Please type a question so I can help you."))
        yield history, ""
        return

    result = bot.ask(message)
    meta = (
        f"\n\n_Confidence: {result.confidence:.2f}  •  {result.timestamp}_"
        if result.matched_question
        else f"\n\n_Confidence: {result.confidence:.2f}  •  {result.timestamp}_"
    )
    full_reply = result.answer + meta

    history.append((message, ""))
    partial = ""
    for ch in full_reply:
        partial += ch
        history[-1] = (message, partial)
        # Typing animation
        time.sleep(0.005)
        yield history, ""


def clear_chat():
    return [], ""


def show_analytics():
    return bot.analytics_json(top_n=5)


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CUSTOM_CSS,
                   title="FAQ Chatbot") as demo:
        gr.Markdown(
            """
            # 🤖 FAQ Chatbot 
            Ask me about **AI, Python, Machine Learning, College FAQs, Programming** and more.
            """
        )

        chatbot = gr.Chatbot(elem_id="chatbot", label="FAQ Assistant",
                             show_copy_button=True, bubble_full_width=False)

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your question and press Enter…",
                show_label=False, scale=8, autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        with gr.Accordion("📊 Analytics", open=False):
            analytics_box = gr.Code(language="json", label="Usage stats")
            refresh_btn = gr.Button("Refresh analytics")

        gr.Examples(
            examples=[
                "What is Machine Learning?",
                "Who created Python?",
                "Explain TF-IDF",
                "Are scholarships available?",
                "What is a transformer model?",
            ],
            inputs=txt,
        )

        send_btn.click(respond, [txt, chatbot], [chatbot, txt])
        txt.submit(respond, [txt, chatbot], [chatbot, txt])
        clear_btn.click(clear_chat, outputs=[chatbot, txt])
        refresh_btn.click(show_analytics, outputs=analytics_box)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue().launch(share=True, server_name="0.0.0.0")