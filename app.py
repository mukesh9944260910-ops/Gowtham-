import asyncio, base64, io, json, os, random, re, time, wave
from pathlib import Path

import edge_tts
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
OUT = Path("output")
CHANNEL = "புராணம் பேசும்"
GREETING = "தமிழுக்கும் தமிழனுக்கும் வணக்கம்."
SIGNOFF = """ஒரு விஷயம் நம்ம கவனிக்கணும்… புராணங்களையும், கோயில் மரபுகளையும், பக்திப் பாடல்களையும்… ஒன்றாகக் கலக்காமல் புரிஞ்சுக்கிட்டால்தான்… நம்முடைய பாரம்பரிய கதைகளின் உண்மையான அழகு இன்னும் தெளிவாகத் தெரியும்.

[SUMMARY]

இன்னும் இப்படிப்பட்ட மறைந்திருக்கும் கதைகளையும், நம்பிக்கைகளின் பின்னால் இருக்கும் வரலாற்றையும் ஆராய்ந்து பார்க்க… இது புராணம் பேசும். Subscribe பண்ணுங்க. அடுத்த கதையோடு மீண்டும் சந்திப்போம். வணக்கம்."""
SIGNATURE_EXAMPLES = [
    "சின்ன வயசுல இருந்து…",
    "ஆனா… இந்த ஒரு வரிக்குள்ள இவ்வளவு பெரிய விஷயம் இருக்குன்னு நமக்குத் தெரியுமா?",
    "இது ஒரு சாதாரண … மட்டும் இல்ல.",
    "இங்க ஒரு முக்கியமான விஷயம்…",
    "இதை … என்று ஒரு ஆதாரமாக எடுத்துக்கொள்ளக் கூடாது.",
]
SHOT_WORDS = 20  # about 8 seconds of Tamil narration


# ---------- models ----------
class Story(BaseModel):
    youtube_title: str
    description: str
    tags: list[str]
    scenes: list[str]
    summary_line: str


class Meta(BaseModel):
    youtube_title: str
    description: str
    tags: list[str]
    summary_line: str


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
    examples = "\n".join("- " + p for p in SIGNATURE_EXAMPLES)
    seed = random.randint(1000, 9999)
    return f"""You are the script writer of the Tamil YouTube channel "Puranam Pesum" (புராணம் பேசும்).
Write a complete voice-over script in Tamil script for the topic: "{title}". Length: about {minutes * 120} words (a {minutes} minute voice-over at spoken Tamil pace).

CHANNEL IDENTITY: spoken/colloquial Tamil (not too formal, not too slang). The channel explores puranas, temple traditions, devotional songs and beliefs and their background, and clearly separates legend from history. Never present myth as fact.
NARRATOR: "a knowledgeable friend", not a scholar showing off; a friend who tells the story with curiosity. Tone: calm + curious + respectful. Respect belief but do not ask for blind belief. Never over-dramatic, never mocking.

REQUIRED STRUCTURE (flow through these in order, without writing the headings):
1. HOOK: quote a familiar line / belief / song and create curiosity ("did you know such a big story is inside this?").
2. QUESTION DROP: 2-3 short questions in a row.
3. FOUNDATION: explain the basic idea simply.
4. DEEP DIVE: go deep with literature / temple / traditional sources.
5. NUANCE/CLARITY: a clear honesty moment: "this should not be understood like this", clearly separating legend from fact.
6. EMOTIONAL CORE: make the listener feel the spiritual beauty of the story.
7. CLOSING REFLECTION: start with "ஒரு விஷயம் நம்ம கவனிக்கணும்…" and reflect on the story.
Do NOT write the sign-off; the app adds the fixed sign-off after your script. Give the story-specific one-line summary separately in summary_line (one Tamil sentence).

LANGUAGE RULES:
- Short sentences, one idea per sentence. Use "…" (ellipsis) often for pauses and suspense.
- Frequent rhetorical questions. Use parallel repetition (e.g. "ஒன்பது கிரகங்கள்… ஒன்பது விதமான தாக்கங்கள்…").
- Minimum English words (only unavoidable technical terms).
- No headings, no scene labels, no [Visual] cues, no stage directions. Clean voice-over text only.
- The very first line must be exactly: {GREETING}

SIGNATURE PHRASES (variation seed {seed}): this channel has recurring "signature phrases" in a friendly, curious, honest spoken-Tamil voice. Here are only EXAMPLES of the kind of phrase, to show the spirit:
{examples}
Do NOT copy these examples word for word, and do not reuse phrases from earlier scripts. For THIS story, invent 3-4 brand-new signature phrases of the same spirit (a personal memory hook, a "did you know" question, a "this is not just an ordinary X" line, an "important point here" line, a caution line), tied to this story's own characters, place and theme. Repeat each of your new phrases 2-3 times across the script so they feel like this story's signature. Every story must end up with different signature phrases.

Split the voice-over into 6-10 consecutive parts ("scenes"); together they form the full script.
Also give: youtube_title (catchy Tamil, under 70 chars), description (Tamil, 150-250 words, with hashtags), tags (15-20, Tamil and English mix)."""


def meta_prompt(script):
    return f"""This is the Tamil voice-over script for a video on the channel "Puranam Pesum".
Write: youtube_title (catchy Tamil, under 70 chars), description (Tamil, 150-250 words, short summary + hashtags), tags (15-20, Tamil and English mix), summary_line (one Tamil sentence summarising this story).
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


def ensure_intro(scenes):
    scenes = [x for x in scenes if x.strip()]
    if scenes and "தமிழுக்கும் தமிழனுக்கும் வணக்கம்" not in scenes[0][:80]:
        scenes[0] = GREETING + " " + scenes[0]
    return scenes


def make_story(title, minutes, own_script):
    if own_script.strip():
        meta = call_json(meta_prompt(own_script), Meta, tokens=4000)
        scenes = chunk_scenes(own_script)
        has_signoff = "இது புராணம் பேசும்" in own_script
    else:
        meta = call_json(story_prompt(title, minutes), Story)
        scenes = meta["scenes"]
        has_signoff = False
    scenes = ensure_intro(scenes)
    if not has_signoff:
        scenes.append(SIGNOFF.replace("[SUMMARY]", meta["summary_line"]))
    return {
        "youtube_title": meta["youtube_title"],
        "description": meta["description"],
        "tags": meta["tags"],
        "scenes": scenes,
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
minutes = st.slider("Video length (minutes)", 3, 15, 7)
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
