import os
import time
import json
import streamlit as st

# Load API key from Streamlit secrets or environment variable
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OpenAI API key. Please set it in Streamlit Secrets.")
    st.stop()
    
# --- OpenAI client with graceful fallbacks (new SDK and legacy) ---
CLIENT_MODE = None
client = None

try:
    # New SDK style
    from openai import OpenAI
    client = OpenAI()
    CLIENT_MODE = "new"
except Exception:
    try:
        # Legacy style
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        CLIENT_MODE = "legacy"
    except Exception as e:
        CLIENT_MODE = None

# -------- Helpers --------
def call_llm(model, messages, temperature=0.7, max_tokens=400):
    """
    Works with both new and legacy OpenAI SDKs.
    messages = [{"role":"system"/"user"/"assistant","content":"..."}]
    """
    if CLIENT_MODE == "new":
        # Prefer chat.completions for broad compatibility
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    elif CLIENT_MODE == "legacy":
        # Legacy (pre-1.0) fallback
        out = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()
    else:
        raise RuntimeError("OpenAI client not initialized. Check your installation and OPENAI_API_KEY.")


def default_system_prompt(name, role, culture):
    return f"""You are {name}, a negotiation agent. Role: {role}.
Cultural profile: {culture}.
Goal: reach a mutually acceptable agreement while following your constraints and style.
Be concise, one turn at a time. Do not invent tools. Avoid repeating yourself.
If you reach a deal, explicitly include the line: AGREEMENT REACHED: <one-sentence summary>.
If no deal is possible, include: NO DEAL: <reason>."""


def render_chat(transcript):
    for turn in transcript:
        with st.chat_message(turn["speaker"]):
            st.markdown(turn["text"])


def detect_stop(text):
    t = text.upper()
    if "AGREEMENT REACHED:" in t:
        return "agreement"
    if "NO DEAL:" in t:
        return "no_deal"
    return None


def make_agent_context(sys_prompt, brief):
    # Order: system, then brief from "user" to prime the agent
    return [{"role": "system", "content": sys_prompt},
            {"role": "user", "content": brief}]


def build_turn_messages(agent_ctx, partner_last_utterance):
    # Agent sees its own system+brief + partner message as "user"
    msgs = list(agent_ctx)  # copy
    if partner_last_utterance:
        msgs.append({"role": "user", "content": partner_last_utterance})
    return msgs


def export_transcript(transcript, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


# ----------------- UI -----------------
st.set_page_config(page_title="SD25 | Bot-to-Bot Milestone 1", page_icon="🤝", layout="wide")
st.title("🤝 Cross-Cultural Negotiation — Milestone 1 (Bot ↔ Bot)")

with st.sidebar:
    st.subheader("Run Settings")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_tokens = st.slider("Max tokens/turn", 100, 1000, 400, 50)
    max_turns = st.number_input("Max turns (total messages, both bots)", min_value=2, max_value=50, value=12, step=2)
    delay_s = st.slider("Delay between turns (sec)", 0.0, 2.0, 0.0, 0.1)

    st.markdown("---")
    st.subheader("Agent A")
    a_name = st.text_input("A Name", value="Aiko")
    a_role = st.text_input("A Role", value="Seller for premium sensor modules")
    a_culture = st.text_area("A Cultural Profile", value="High-context, relationship-first; prefers indirect concessions and face-saving.")
    a_sys = st.text_area("A System Prompt (optional)", value="", placeholder="Leave blank to auto-generate from Role/Culture.")
    st.markdown("---")
    st.subheader("Agent B")
    b_name = st.text_input("B Name", value="Blake")
    b_role = st.text_input("B Role", value="Procurement lead seeking volume discount")
    b_culture = st.text_area("B Cultural Profile", value="Low-context, direct; values clear terms, timelines, and price transparency.")
    b_sys = st.text_area("B System Prompt (optional)", value="", placeholder="Leave blank to auto-generate from Role/Culture.")

    st.markdown("---")
    st.subheader("Scenario")
    brief = st.text_area(
        "Shared Negotiation Brief (paste from your SD25 docs)",
        value=(
            "Context: Procurement of 10,000 sensor modules for Q4.\n"
            "Constraints: Seller wants ≥$48/unit and 50% upfront. Buyer target ≤$42/unit, net-30.\n"
            "Objective: Find terms acceptable to both in ≤6 messages each.\n"
            "Output: clear, concise proposals; stop when agreement or stalemate."
        ),
        height=160
    )

    run_btn = st.button("▶️ Start Run", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)

# Session state
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dialogue")
    render_chat(st.session_state.transcript)

with col2:
    st.subheader("Controls & Export")

    if reset_btn:
        st.session_state.transcript = []
        st.session_state.running = False
        st.success("Session reset.")

    # Prepare system prompts
    a_system = a_sys.strip() or default_system_prompt(a_name, a_role, a_culture)
    b_system = b_sys.strip() or default_system_prompt(b_name, b_role, b_culture)

    # Agent contexts (system + brief)
    a_ctx = make_agent_context(a_system, f"Negotiation brief:\n{brief}")
    b_ctx = make_agent_context(b_system, f"Negotiation brief:\n{brief}")

    # Starter: who speaks first?
    first_speaker = st.selectbox("First speaker", [a_name, b_name], index=0)

    if run_btn:
        if CLIENT_MODE is None:
            st.error("OpenAI client not initialized. Ensure `pip install openai` and set OPENAI_API_KEY.")
        else:
            st.session_state.running = True
            st.toast("Run started", icon="✅")

            # Initialize last utterances
            last_a = None
            last_b = None

            # If B speaks first, seed with empty A
            next_speaker = first_speaker

            # Clear previous transcript
            st.session_state.transcript = []

            # Loop turns
            for t in range(max_turns):
                if next_speaker == a_name:
                    # A responds to B's last
                    msgs = build_turn_messages(a_ctx, last_b)
                    try:
                        a_out = call_llm(model, msgs, temperature, max_tokens)
                    except Exception as e:
                        st.error(f"LLM error (A): {e}")
                        break

                    st.session_state.transcript.append({"speaker": "assistant", "who": a_name, "text": a_out})
                    with st.chat_message("assistant"):
                        st.markdown(f"**{a_name}:** {a_out}")

                    stop = detect_stop(a_out)
                    if stop:
                        st.info(f"Stop condition: {stop.upper()}")
                        break

                    last_a = a_out
                    next_speaker = b_name

                else:  # B's turn
                    msgs = build_turn_messages(b_ctx, last_a)
                    try:
                        b_out = call_llm(model, msgs, temperature, max_tokens)
                    except Exception as e:
                        st.error(f"LLM error (B): {e}")
                        break

                    st.session_state.transcript.append({"speaker": "assistant", "who": b_name, "text": b_out})
                    with st.chat_message("assistant"):
                        st.markdown(f"**{b_name}:** {b_out}")

                    stop = detect_stop(b_out)
                    if stop:
                        st.info(f"Stop condition: {stop.upper()}")
                        break

                    last_b = b_out
                    next_speaker = a_name

                if delay_s > 0:
                    time.sleep(delay_s)

            st.session_state.running = False

    # Export buttons
    if st.session_state.transcript:
        ts = int(time.time())
        json_path = f"transcript_{ts}.json"
        txt_path = f"transcript_{ts}.txt"

        if st.button("💾 Export JSON", use_container_width=True):
            export_transcript(st.session_state.transcript, json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button("Download transcript JSON", f, file_name=json_path, mime="application/json", use_container_width=True)

        if st.button("📄 Export Plain Text", use_container_width=True):
            text_dump = []
            for turn in st.session_state.transcript:
                text_dump.append(f"{turn['who']}: {turn['text']}")
            text_str = "\n\n".join(text_dump)
            st.download_button("Download transcript TXT", text_str, file_name=txt_path, mime="text/plain", use_container_width=True)

