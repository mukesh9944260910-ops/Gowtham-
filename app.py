import asyncio, base64, io, json, re, time, wave
from pathlib import Path

import os

import edge_tts
from google import genai
from google.genai import types
import streamlit as st

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


VOICES = {
    "Valluvar (India)": "ta-IN-ValluvarNeural",
    "Kumar (Sri Lanka)": "ta-LK-KumarNeural",
    "Anbu (Singapore)": "ta-SG-AnbuNeural",
    "Surya (Malaysia)": "ta-MY-SuryaNeural",
}


def make_voice(texts, path, voice, rate, pitch):
    async def run():
        parts = []
        for t in texts:
            tmp = path.with_suffix(".part.mp3")
            await edge_tts.Communicate(t, voice, rate=rate, pitch=pitch).save(str(tmp))
            parts.append(tmp.read_bytes())
            tmp.unlink()
        path.write_bytes(b"".join(parts))

    asyncio.run(run())


GEMINI_VOICES = {
    "Charon (informative)": "Charon",
    "Orus (firm)": "Orus",
    "Iapetus (clear)": "Iapetus",
    "Rasalgethi (informative)": "Rasalgethi",
    "Algenib (gravelly, deep)": "Algenib",
    "Alnilam (firm)": "Alnilam",
    "Fenrir (excitable)": "Fenrir",
}
STYLE = "deep, calm, devotional storyteller voice, narrating slowly with warmth and emotion"


def gemini_voice(texts, path, voice, model, style):
    client = genai.Client()
    frames, params = [], None
    for t in texts:
        for attempt in range(3):
            try:
                it = client.interactions.create(
                    model=model,
                    input=[{"type": "user_input", "content": [{
                        "type": "text", "text": t,
                        "annotations": [{"type": "speech_metadata", "style": style}],
                    }]}],
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": voice}]},
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(25)  # free tier rate limit, wait and retry
        with wave.open(io.BytesIO(base64.b64decode(it.output_audio.data))) as w:
            params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
            frames.append(b"\x00" * params.sampwidth * params.framerate // 2)  # 0.5s gap
    with wave.open(str(path), "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))


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
    engine = st.radio("Voice engine", ["Gemini (best quality)", "Edge (backup)"])
    if engine.startswith("Gemini"):
        vname = st.selectbox("Voice", list(GEMINI_VOICES))
        gmodel = st.selectbox("Model", ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"])
        style = st.text_input("Style", STYLE)
        ext = "wav"

        def synth(texts, path):
            gemini_voice(texts, path, GEMINI_VOICES[vname], gmodel, style)
    else:
        vname = st.selectbox("Voice", list(VOICES))
        speed = st.slider("Speed", -30, 10, -10, format="%d%%")
        pitch = st.slider("Pitch", -30, 10, -8, format="%dHz")
        ext = "mp3"

        def synth(texts, path):
            make_voice(texts, path, VOICES[vname], f"{speed:+d}%", f"{pitch:+d}Hz")

    c1, c2 = st.columns(2)
    if c1.button("Test (scene 1)"):
        with st.spinner("Test voice..."):
            t = folder / f"test.{ext}"
            synth([data["scenes"][0]["narration"]], t)
            st.audio(str(t))
    if c2.button("Full voice", type="primary"):
        with st.spinner("Voice generate aaguthu, konjam neram aagum..."):
            (folder / "script.txt").write_text(
                "\n\n".join(x["narration"] for x in data["scenes"]), encoding="utf-8")
            out = folder / f"voice.{ext}"
            synth([x["narration"] for x in data["scenes"]], out)
            st.session_state.audio = out
    if st.session_state.get("audio"):
        a_path = st.session_state.audio
        st.audio(str(a_path))
        st.download_button("Download voice", a_path.read_bytes(), file_name=a_path.name,
                           mime="audio/wav" if a_path.suffix == ".wav" else "audio/mpeg")
        st.download_button("Download story (json)", (folder / "story.json").read_bytes(),
                           file_name="story.json", mime="application/json")
