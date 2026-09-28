import asyncio, json, re, time
from pathlib import Path

import os

import edge_tts
from google import genai
from google.genai import types
import streamlit as st

VOICE = "ta-IN-ValluvarNeural"  # Tamil male voice
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
OUT = Path("output")

PROMPT = """You are the scriptwriter for a Tamil YouTube channel "Puranam Pesum" that tells puranam stories.
Write a gripping, emotional storytelling script in Tamil (Tamil script) for the title: "{title}".
Length: about {minutes} minutes of narration. Start with a strong hook, keep a devotional and respectful tone,
end with the moral and a line asking viewers to like, share and subscribe.
Split it into scenes. Return ONLY valid JSON, no markdown fences:
{{
 "youtube_title": "catchy Tamil title (under 70 chars)",
 "description": "Tamil description, 150-250 words, with a short summary and hashtags",
 "tags": ["15-20 tags mixing Tamil and English"],
 "scenes": [
  {{"narration": "Tamil narration for this scene (60-120 words)",
    "visual_prompt": "English image prompt: cinematic, traditional Indian mythological art style, describing the scene"}}
 ]
}}"""


def generate(title, minutes):
    client = genai.Client()  # uses GEMINI_API_KEY
    resp = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(title=title, minutes=minutes),
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    text = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip()
    return json.loads(text)


def make_voice(text, path):
    asyncio.run(edge_tts.Communicate(text, VOICE, rate="-5%").save(str(path)))


st.set_page_config(page_title="Puranam Pesum Studio", page_icon="🪔")
st.title("🪔 Puranam Pesum Studio")

title = st.text_input("Story title", placeholder="e.g. Markandeyan and Yama")
minutes = st.slider("Video length (minutes)", 3, 15, 6)

if st.button("Create story", type="primary") and title:
    with st.spinner("Script ezhudhuren..."):
        st.session_state.data = generate(title, minutes)
        safe = re.sub(r"\W+", "_", title)[:30]
        st.session_state.folder = OUT / f"{int(time.time())}_{safe}"
        st.session_state.folder.mkdir(parents=True, exist_ok=True)
        st.session_state.audio = None

data = st.session_state.get("data")
if data:
    folder = st.session_state.folder
    (folder / "story.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    st.header("YouTube details")
    data["youtube_title"] = st.text_input("Title", data["youtube_title"])
    data["description"] = st.text_area("Description", data["description"], height=220)
    st.text_area("Tags (comma separated)", ", ".join(data["tags"]))

    st.header("Script + visuals")
    for i, s in enumerate(data["scenes"], 1):
        with st.expander(f"Scene {i}", expanded=i == 1):
            s["narration"] = st.text_area("Narration", s["narration"], key=f"n{i}", height=140)
            st.code(s["visual_prompt"], language=None)

    st.header("Male voice")
    if st.button("Generate voice"):
        with st.spinner("Voice generate aaguthu..."):
            full = "\n\n".join(s["narration"] for s in data["scenes"])
            (folder / "script.txt").write_text(full, encoding="utf-8")
            mp3 = folder / "voice.mp3"
            make_voice(full, mp3)
            st.session_state.audio = mp3
    if st.session_state.get("audio"):
        st.audio(str(st.session_state.audio))
        st.download_button("Download voice (mp3)", st.session_state.audio.read_bytes(),
                           file_name="voice.mp3", mime="audio/mpeg")
        st.download_button("Download story (json)", (folder / "story.json").read_bytes(),
                           file_name="story.json", mime="application/json")
