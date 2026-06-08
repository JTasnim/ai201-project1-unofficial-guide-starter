"""
app.py — Gradio Web Interface
The Unofficial Guide: UC Berkeley CS Professor Reviews

Run with: python app.py
Then open: http://localhost:7860

Interface:
- Input:  question textbox
- Output: answer textbox + sources textbox
- Button: Ask / submit on Enter
"""

import gradio as gr
from generate import generate_answer


def handle_query(question: str):
    """
    Called when the user submits a question.
    Returns (answer, sources_text) for the two output textboxes.
    """
    question = question.strip()

    if not question:
        return "Please enter a question.", ""

    try:
        result = generate_answer(question)
        answer = result["answer"]
        sources_text = "\n".join(f"• {s}" for s in result["sources"])
        return answer, sources_text

    except Exception as e:
        return f"Error: {str(e)}\n\nMake sure your GROQ_API_KEY is set in your .env file.", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="The Unofficial Guide — UC Berkeley CS") as demo:

    gr.Markdown("""
    # 🎓 The Unofficial Guide: UC Berkeley CS Professors
    Ask any question about UC Berkeley CS professors and courses.
    Answers are grounded in real student reviews — no hallucination.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder='e.g. "What do students say about CS61B workload?" or "Is CS70 hard?"',
                lines=2,
            )
            ask_button = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                label="Answer (grounded in student reviews)",
                lines=10,
                interactive=False,
            )
        with gr.Column(scale=1):
            sources_output = gr.Textbox(
                label="Retrieved from",
                lines=10,
                interactive=False,
            )

    gr.Markdown("""
    ---
    **Example questions to try:**
    - What do students say about exam difficulty in CS189?
    - Which professors are known for being good at office hours?
    - How is the workload for CS61B?
    - What do students think of Dawn Song's CS161?
    - Is CS170 hard? (tests an out-of-scope question)
    """)

    # Wire up interactions
    ask_button.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )

    # Also submit on Enter key
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    print("Starting The Unofficial Guide...")
    print("Open http://localhost:7860 in your browser")
    demo.launch()