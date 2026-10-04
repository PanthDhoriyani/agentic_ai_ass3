"""
Generates the architecture interaction-flow diagram for CogniAssist Agent.
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)

    # Title
    ax.text(50, 66, "CogniAssist: Agent Memory-Tool Architecture & Interaction Flow", 
            ha='center', va='center', fontsize=14, fontweight='bold', color='#1A237E')

    # Color palette
    c_user = "#E3F2FD"      # Light Blue
    c_adapt = "#FFF3E0"     # Light Orange
    c_ltm = "#E8F5E9"       # Light Green
    c_stm = "#EDE7F6"       # Light Purple
    c_agent = "#E0F2F1"     # Light Teal
    c_tools = "#FCE4EC"     # Light Pink
    c_out = "#ECEFF1"       # Light Gray

    # Helper function for styled boxes
    def draw_box(x, y, w, h, text, bg, border='#37474F', title=None):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                     linewidth=1.5, edgecolor=border, facecolor=bg)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 2.5, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#263238')
            ax.text(x + w/2, y + h/2 - 1.2, text, ha='center', va='center', fontsize=8.5, color='#37474F')
        else:
            ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#263238')

    # Helper for arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#37474F", lw=1.5, shrinkA=3, shrinkB=3))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 1.2, label, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#455A64',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    # Boxes
    # 1. User Input
    draw_box(38, 54, 24, 7, "Query / Dialogue Turn\n(Session-Aware)", c_user, title="1. User Input")

    # 2. Feedback Adaptation Detector
    draw_box(38, 41, 24, 7.5, "Intent Classification\nCorrection vs New Fact", c_adapt, title="2. Adaptation Loop")

    # 3. Long Term Memory (FAISS)
    draw_box(4, 27, 26, 9.5, "Dense Vector Index (Cosine)\nMetadata: active / superseded\nCross-Session Recall", c_ltm, title="3. Long-Term Memory (FAISS)")

    # 4. Short-Term Memory (STM)
    draw_box(70, 27, 26, 9.5, "Conversation Buffer\nSliding Window (Max 10 Turns)\nSession Isolation", c_stm, title="4. Short-Term Memory (Buffer)")

    # 5. Core Reasoning Engine
    draw_box(36, 26, 28, 10.5, "Tool Selection & Planning\nPrompt Synthesis & Reflection\nContext Merger", c_agent, title="5. Autonomous Agent Core")

    # 6. External Tools
    draw_box(30, 11, 40, 9.5, "Safe Calculator | Python Sandbox | Live Web Search", c_tools, title="6. Tool / Function Calling Layer")

    # 7. Final Response
    draw_box(36, 1, 28, 6.5, "Contextual Answer + Tool Output\nMemory Confirmation", c_out, title="7. Synthesized Response")

    # Interconnections
    draw_arrow(50, 54, 50, 48.5, "")
    draw_arrow(38, 44.5, 17, 36.5, "Correction")
    draw_arrow(17, 27, 36, 31, "Top-k Vectors")
    draw_arrow(50, 41, 50, 36.5, "Query")
    draw_arrow(62, 44.5, 83, 36.5, "Turn Log")
    draw_arrow(70, 31, 64, 31, "Recent Turns")
    draw_arrow(50, 26, 50, 20.5, "Tool Dispatch")
    draw_arrow(50, 11, 50, 7.5, "Tool Output")

    plt.tight_layout()
    plt.savefig("architecture_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Architecture diagram created: architecture_diagram.png")

if __name__ == "__main__":
    generate_architecture_diagram()
