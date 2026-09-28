import asyncio, base64, io, json, os, re, time, wave
from pathlib import Path

import edge_tts
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
OUT = Path("output")
CHANNEL = "புராணம் பேசும்"
GREETING = "தமிழுக்கும் தமிழனுக்கும் வணக்கம்!"
OUTRO = f"இந்தக் கதை உங்களுக்குப் பிடித்திருந்தால், {CHANNEL} சேனலை மறக்காமல் சப்ஸ்க்ரைப் செய்யுங்கள். லைக் செய்து, ஷேர் செய்யுங்கள். நன்றி, வணக்கம்!"
SHOT_WORDS = 20  # about 8 seconds of Tamil narration


# ---------- models ----------
class Story(BaseModel):
    youtube_title: str
    description: str
    tags: list[str]
    scenes: list[str]


class Meta(BaseModel):
    youtube_title: str
    description: str
    tags: list[str]


class Shots(BaseModel):
    style_guide: str
    prompts: list[str]


# ---------- gemini helpers ----------
def call_json(prompt, schema, tokens=16000):
    client = genai.Client()  # uses GEMINI_API_KEY
    last = None
    for _ in range(3):
        try:
            r = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    max_output_tokens=tokens,
                ),
            )
            if r.parsed:
                return r.parsed.model_dump()
            return json.loads(re.sub(r"^```(?:json)?|```$", "", r.text.strip(), flags=re.M))
        except Exception as e:
            last = e
            time.sleep(4)
    raise last


def story_prompt(title, minutes):
    return f"""You are the scriptwriter of the Tamil YouTube channel "{CHANNEL}" (Puranam Pesum) that tells puranam stories.
Write a gripping storytelling narration in Tamil script for the title: "{title}". Length: about {minutes * 120} words.
Rules:
- The very first line must be exactly: {GREETING} Then flow smoothly into a strong hook.
- Devotional, respectful, emotional storytelling in simple spoken Tamil with short sentences (it will be voiced).
- Ending: after the moral of the story, smoothly and warmly invite viewers to subscribe to the "{CHANNEL}" channel, like and share, then close with a warm goodbye. It must feel natural, not abrupt.
- No headings, no scene labels, no stage directions, no English words inside the narration.
Split the narration into 5-8 consecutive parts ("scenes"); together they form the full script.
Also give: youtube_title (catchy Tamil, under 70 chars), description (Tamil, 150-250 words, with hashtags), tags (15-20, Tamil and English mix)."""


def meta_prompt(script):
    return f"""This is the Tamil narration script for a video on the channel "{CHANNEL}".
Write: youtube_title (catchy Tamil, under 70 chars), description (Tamil, 150-250 words, short summary + hashtags), tags (15-20, Tamil and English mix).
SCRIPT:
{script[:6000]}"""


def chunk_scenes(text, words=90):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) > 1:
        return paras
    sents = re.split(r"(?<=[.!?।])\s+", text.strip())
    scenes, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= words:
            scenes.append(" ".join(cur))
            cur = []
    if cur:
        scenes.append(" ".join(cur))
    return scenes


def ensure_intro_outro(scenes):
    scenes = [s for s in scenes if s.strip()]
    if not scenes:
        return scenes
    if "வணக்கம்" not in scenes[0][:80]:
        scenes[0] = GREETING + " " + scenes[0]
    if CHANNEL not in scenes[-1]:
        scenes[-1] = scenes[-1].rstrip() + " " + OUTRO
    return scenes


def make_story(title, minutes, own_script):
    if own_script.strip():
        meta = call_json(meta_prompt(own_script), Meta, tokens=4000)
        scenes = chunk_scenes(own_script)
    else:
        d = call_json(story_prompt(title, minutes), Story)
        meta, scenes = d, d["scenes"]
    return {
        "youtube_title": meta["youtube_title"],
        "description": meta["description"],
        "tags": meta["tags"],
        "scenes": ensure_intro_outro(scenes),
    }


# ---------- visuals for Google Flow ----------
def split_shots(text, target=SHOT_WORDS):
    sents = re.split(r"(?<=[.!?।])\s+", text.strip())
    shots, cur = [], []
    for s in sents:
        w = s.split()
        if not w:
            continue
        if cur and len(cur) + len(w) > target + 6:
            shots.append(" ".join(cur))
            cur = []
        cur += w
        if len(cur) >= target:
            shots.append(" ".join(cur))
            cur = []
    if cur:
        if shots and len(cur) < 8:
            shots[-1] += " " + " ".join(cur)
        else:
            shots.append(" ".join(cur))
    return shots


def shot_prompt(title, batch, start, style_guide):
    numbered = "\n".join(f"{start + i + 1}. {t}" for i, t in enumerate(batch))
    if style_guide:
        guide = f"Use exactly this style guide for every shot: {style_guide}\nReturn the same style_guide text back."
    else:
        guide = (
            "First write a style_guide (English, 40-60 words): the art style (rich cinematic look inspired by "
            "traditional Indian temple murals and paintings), colour palette, lighting, and fixed visual "
            "descriptions of the recurring characters (age, clothing, ornaments, skin tone) so they look the same in every shot."
        )
    return f"""You write shot prompts for Google Flow (Veo video generation) for a Tamil puranam YouTube video titled "{title}".
{guide}
Then, for each numbered narration line below (each is about 8 seconds of voiceover), write ONE English prompt describing what to show:
subject, action, setting, camera movement, lighting, mood. Every prompt must be self-contained (repeat the key character descriptions, never say "same as before"),
describe one continuous 8-second shot in 16:9, with no text, no subtitles, no dialogue, no logos. Depict deities respectfully.
Return exactly {len(batch)} prompts, in order.
NARRATION LINES:
{numbered}"""


def make_shots(title, scenes, progress=None):
    lines = [s for sc in scenes for s in split_shots(sc)]
    out, guide, B = [], "", 12
    for i in range(0, len(lines), B):
        batch = lines[i : i + B]
        r = call_json(shot_prompt(title, batch, i, guide), Shots, tokens=8000)
        guide = guide or r["style_guide"]
        ps = (r["prompts"] + ["(prompt missing, generate again)"] * len(batch))[: len(batch)]
        out += ps
        if progress:
            progress.progress(min(1.0, (i + B) / len(lines)))
    return [
        {"text": t, "prompt": f"{p}\n\nStyle: {guide}"} for t, p in zip(lines, out)
    ]


# ---------- voice ----------
VOICES = {
    "Valluvar (India)": "ta-IN-ValluvarNeural",
    "Kumar (Sri Lanka)": "ta-LK-KumarNeural",
    "Anbu (Singapore)": "ta-SG-AnbuNeural",
    "Surya (Malaysia)": "ta-MY-SuryaNeural",
}
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
            frames.append(b"\x00" * (params.sampwidth * params.nchannels * params.framerate // 2))
    with wave.open(str(path), "wb") as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))


# ---------- UI ----------
st.set_page_config(page_title="Puranam Pesum Studio", page_icon="🪔")
st.title("🪔 Puranam Pesum Studio")

title = st.text_input("Story title", placeholder="e.g. Markandeyan and Yama")
minutes = st.slider("Video length (minutes)", 3, 15, 6)
with st.expander("Ennoda own script use panna (optional)"):
    own = st.text_area("Tamil script paste pannunga", height=200)

if st.button("Create story", type="primary") and (title or own.strip()):
    with st.spinner("Script ready aaguthu..."):
        st.session_state.data = make_story(title or "Puranam story", minutes, own)
        st.session_state.title = title or "Puranam story"
        st.session_state.sid = str(int(time.time()))
        safe = re.sub(r"\W+", "_", title)[:30] or "story"
        st.session_state.folder = OUT / f"{st.session_state.sid}_{safe}"
        st.session_state.folder.mkdir(parents=True, exist_ok=True)
        st.session_state.audio = None
        st.session_state.shots = None

data = st.session_state.get("data")
if data:
    folder, sid = st.session_state.folder, st.session_state.sid

    st.header("YouTube details")
    data["youtube_title"] = st.text_input("Title", data["youtube_title"], key=f"t{sid}")
    data["description"] = st.text_area("Description", data["description"], height=220, key=f"d{sid}")
    st.text_area("Tags (comma separated)", ", ".join(data["tags"]), key=f"g{sid}")

    st.header("Script")
    for i, s in enumerate(data["scenes"]):
        data["scenes"][i] = st.text_area(f"Part {i + 1}", s, key=f"n{sid}_{i}", height=140)
    (folder / "story.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    st.header("Visual prompts (Google Flow)")
    st.caption("Ovvoru prompt um oru 8-second shot ku. Script edit pannina piragu generate pannunga.")
    if st.button("Generate visual prompts"):
        bar = st.progress(0.0)
        with st.spinner("Prompts ezhudhuren..."):
            st.session_state.shots = make_shots(st.session_state.title, data["scenes"], bar)
    shots = st.session_state.get("shots")
    if shots:
        st.success(f"{len(shots)} shots ready (~{len(shots) * 8 // 60} min {len(shots) * 8 % 60} sec)")
        for i, sh in enumerate(shots, 1):
            with st.expander(f"Shot {i}"):
                st.caption(sh["text"])
                st.code(sh["prompt"], language=None)
        allp = "\n\n".join(f"Shot {i}\n{sh['prompt']}" for i, sh in enumerate(shots, 1))
        st.download_button("Download all prompts (txt)", allp, file_name="flow_prompts.txt")

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
    if c1.button("Test (part 1)"):
        with st.spinner("Test voice..."):
            t = folder / f"test.{ext}"
            synth([data["scenes"][0]], t)
            st.audio(str(t))
    if c2.button("Full voice", type="primary"):
        with st.spinner("Voice generate aaguthu, konjam neram aagum..."):
            out = folder / f"voice.{ext}"
            synth(data["scenes"], out)
            st.session_state.audio = out
    if st.session_state.get("audio"):
        a_path = st.session_state.audio
        st.audio(str(a_path))
        st.download_button("Download voice", a_path.read_bytes(), file_name=a_path.name,
                           mime="audio/wav" if a_path.suffix == ".wav" else "audio/mpeg")
