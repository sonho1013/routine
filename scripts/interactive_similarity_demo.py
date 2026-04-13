"""
交互式相似度 Demo — 输入一条 fact 文本，实时查看 ChromaDB 中 top-5 最近邻

用于架构合理性汇报现场演示，让领导亲手体验 text fact 的区分能力。

启动方式:
    python scripts/interactive_similarity_demo.py
    # 浏览器打开 http://localhost:8501

依赖: pip install gradio
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gradio as gr
except ImportError:
    print("请安装 gradio: pip install gradio")
    sys.exit(1)

from panoramix_core.store.fact_store_chroma import FactStoreChroma
from panoramix_core.models.fact_enums import FactType

# ── 配置 ──
USERNAME = "Mary"
TOP_K = 5


def get_store():
    """获取 FactStoreChroma 实例"""
    return FactStoreChroma(USERNAME)


def query_similar(input_text: str, top_k: int = TOP_K):
    """
    查询与输入文本最相似的 facts。

    Returns:
        HTML 格式的结果表
    """
    if not input_text.strip():
        return "<p style='color: gray;'>请输入一条动作描述文本</p>"

    store = get_store()
    try:
        results = store.query_similar(USERNAME, input_text, n_results=top_k + 1)
    except Exception as e:
        return f"<p style='color: red;'>查询失败: {e}</p>"
    finally:
        store.close()

    if not results:
        return "<p style='color: gray;'>数据库中未找到 facts。请先运行学习流程。</p>"

    # Build HTML table
    html = """
    <style>
        .sim-table { border-collapse: collapse; width: 100%; font-size: 14px; }
        .sim-table th { background: #1565C0; color: white; padding: 10px; text-align: left; }
        .sim-table td { padding: 8px 10px; border-bottom: 1px solid #eee; }
        .sim-table tr:hover { background: #f5f5f5; }
        .sim-high { color: #2E7D32; font-weight: bold; }
        .sim-mid { color: #E65100; font-weight: bold; }
        .sim-low { color: #C62828; font-weight: bold; }
        .bar-bg { background: #E0E0E0; border-radius: 4px; height: 20px; width: 200px; display: inline-block; }
        .bar-fill { height: 20px; border-radius: 4px; display: inline-block; }
        .type-badge { padding: 2px 8px; border-radius: 12px; font-size: 12px; }
        .type-habit { background: #E8F5E9; color: #2E7D32; }
        .type-pref { background: #E3F2FD; color: #1565C0; }
    </style>
    <table class="sim-table">
        <tr>
            <th>#</th>
            <th>Fact Text</th>
            <th>Type</th>
            <th>Similarity</th>
            <th></th>
        </tr>
    """

    for i, r in enumerate(results[:top_k]):
        text = r.get("text", "N/A")
        distance = r.get("distance", 0)
        similarity = (1 - distance) * 100
        fact_type = r.get("type", "PREF")

        # Skip exact self-match
        if similarity > 99.9 and text.strip().lower() == input_text.strip().lower():
            continue

        # Color class
        if similarity >= 90:
            css_class = "sim-high"
            bar_color = "#4CAF50"
        elif similarity >= 80:
            css_class = "sim-mid"
            bar_color = "#FF9800"
        else:
            css_class = "sim-low"
            bar_color = "#F44336"

        # Type badge
        type_class = "type-habit" if fact_type == "HABIT" else "type-pref"

        bar_width = max(0, min(200, similarity * 2))

        html += f"""
        <tr>
            <td>{i + 1}</td>
            <td>{text}</td>
            <td><span class="type-badge {type_class}">{fact_type}</span></td>
            <td class="{css_class}">{similarity:.1f}%</td>
            <td>
                <div class="bar-bg">
                    <div class="bar-fill" style="width: {bar_width}px; background: {bar_color};"></div>
                </div>
            </td>
        </tr>
        """

    html += "</table>"

    # Threshold note
    html += """
    <p style="margin-top: 16px; color: #666; font-size: 13px;">
        <b>DBSCAN threshold = 90%</b> — facts above 90% similarity would be clustered together as the same habit.
        <br>This is <b>text cosine similarity only</b>. The full system also factors in context (time, location, vehicle state).
    </p>
    """

    return html


# ── Example inputs ──
EXAMPLES = [
    ["set cabin air conditioning temperature to 22 degrees"],
    ["turned on seat heating to level 2"],
    ["started navigation to work"],
    ["resumed listening to podcast Tech Daily"],
    ["switched to eco driving mode"],
    ["lowered driver window to 50 percent"],
    ["closed all vehicle windows"],
    ["shut down the vehicle engine"],
    ["adjusted media volume to 58 percent"],
    ["turned off cabin air conditioning"],
]


def build_demo():
    with gr.Blocks(
        title="Habit Memory — Similarity Explorer",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # Habit Memory — Text Fact Similarity Explorer

        输入一条用户动作的文本描述，查看数据库中最相似的 facts。
        这个 demo 展示了 embedding 向量化对动作语义的区分能力。

        **试试看：** 修改温度数值（22→24）、换一个动作类型、或输入一个全新的动作。
        """)

        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    label="输入动作描述 (Input action text)",
                    placeholder="e.g., set cabin air conditioning temperature to 22 degrees",
                    lines=2,
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("Search", variant="primary", size="lg")

        output_html = gr.HTML(label="Top-5 Most Similar Facts")

        gr.Examples(
            examples=EXAMPLES,
            inputs=input_text,
            label="示例输入 (click to try)",
        )

        search_btn.click(fn=query_similar, inputs=input_text, outputs=output_html)
        input_text.submit(fn=query_similar, inputs=input_text, outputs=output_html)

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=8501, share=False)
