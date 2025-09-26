#!/usr/bin/env python3
import os
import sys
import time
import json
import requests
from secrets import OPENAI_API_KEY
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI
import random  

API_BASE = "https://api.decentralised.art"
DEFAULT_NUM_VOCAB = 8

def _get_account() -> Account :
    priv = os.getenv("PRIVATE_KEY") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if priv:
        acct = Account.from_key(priv)
        print("Loaded account from PRIVATE_KEY/CLI.")
    else:
        acct = Account.create('KEYSMASH FJAFJKLDSKF7JKFDJ 1530')
        print("Created example local account (ephemeral).")
    print(f"Address: {acct.address}")
    return acct

def _handle_response(r : requests.Response) :
    try:
        data = r.json()
    except json.JSONDecodeError:
        r.raise_for_status()
        return {"raw": r.text}
    if not r.ok:
        print(f" fail: {r.status_code} {data}", file=sys.stderr)
        raise requests.HTTPError(f" fail: {r.status_code} {data}", response=r)
    return data 

def get_nonce(base_url: str, address: str, timeout: float = 10.0) -> str:
    url = f"{base_url}/nonce/{address}"
    r = requests.get(url, headers={"Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict):
        if "nonce" in data:
            return str(data["nonce"])
    raise ValueError(f"Unexpected nonce response shape: {data}")

def post_auth(base_url: str, address: str, message: str, signature: str, timeout: float = 10.0) -> dict:
    """
    Calls /auth
    Returns parsed JSON (expected to include access_token and refresh_token).
    """
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    payload = {
        "address": address,
        "message": message,
        "signature": signature
    }

    r = requests.post(f"{base_url}/auth", headers=headers, json=payload, timeout=timeout)
    return _handle_response(r)

def post_refresh(base_url: str, access_token: str, refresh_token: str, timeout: float = 10.0) -> dict:
    """
    Calls /refresh using headers:
      - Authorization: Bearer <access_token>
      - X-Refresh-Token: <refresh_token>
    Returns parsed JSON (expected to include new access_token and refresh_token).
    """

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "X-Refresh-Token": refresh_token,
    }

    payload={}
    
    r = requests.post(f"{base_url}/refresh", headers=headers, json=payload, timeout=timeout)
    return _handle_response(r)

def post_feature(base_url: str, access_token: str, refresh_token: str, payload: dict, timeout: float = 10.0) -> dict:
    """
    Calls POST /feature with both access and refresh tokens in headers.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "X-Refresh-Token": refresh_token,
    }

    r = requests.post(f"{base_url}/feature", headers=headers, json=payload, timeout=timeout)
    return _handle_response(r)

def generate_feature(user_prompt: str) -> dict:
    """
    Uses OpenAI API to generate a JSON payload for /feature.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_msg = (
        "ROLE:\n"
        "You output ONLY a single valid JSON object describing a DCN Performative Transaction (PT) FEATURE DEFINITION. "
        "Never include prose, code fences, or comments.\n"
        "\n"
        "INTENT:\n"
        "The JSON you return will be used by an off-chain SDK to construct a composite Feature that will later be resolved "
        "by the on-chain Registry and executed by a Runner. Each JSON 'dimension' corresponds to ONE composite subfeature "
        "and its ordered list of transformations (cf. FeatureBase/CallDef in the contracts).\n"
        "\n"
        "REQUIRED JSON SHAPE:\n"
        "{\n"
        "  \"name\": \"<string>\",\n"
        "  \"dimensions\": [\n"
        "    {\n"
        "      \"feature_name\": \"<string>\",   // NAME of an existing Feature in the Registry (e.g., \"pitch\", \"time\", \"duration\"),\n"
        "      \"transformations\": [            // ordered, applied cyclically per Runner::transform\n"
        "        {\"name\": \"<string>\", \"args\": [<uint32>]}  // one arg for each op (see ALLOWED OPS below)\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "HARD CONSTRAINTS:\n"
        "- Output EXACTLY one JSON object; allowed keys ONLY: {name, dimensions, feature_name, transformations, args}.\n"
        "- 'dimensions' is an array; EACH element defines ONE composite subfeature by NAME plus its transformations.\n"
        "- 'transformations' is an array applied IN ORDER; Runner will index them modulo length (cyclic).\n"
        "- ALLOWED transformation names ONLY: [\"add\", \"subtract\"].\n"
        "- For every transformation, 'args' MUST be an array with EXACTLY ONE unsigned integer (uint32). No negatives, no floats.\n"
        "- Use SMALL unsigned integers by default (e.g., 0–127) unless explicitly instructed otherwise.\n"
        "- Keep JSON compact and strictly valid (no trailing commas; no extra fields; no text outside the object).\n"
        "\n"
        "SEMANTICS (match Solidity ITransformation::run signature):\n"
        "- add:     returns x + args[0]\n"
        "- subtract:returns x - args[0]\n"
        "These operate on uint32 indices generated by the Runner; arithmetic must avoid under/overflow.\n"
        "\n"
        "FEATURES & COMPOSITION:\n"
        "- 'feature_name' may be ANY valid Feature name resolvable by the Registry at run time, including higher-level "
        "  composites (e.g., a feature that itself references \"pitch\" and \"time\").\n"
        "- Scalar features (like terminal \"pitch\", \"time\", \"duration\") have no subfeatures; your JSON builds COMPOSITE "
        "  features by listing such subfeatures in 'dimensions'.\n"
        "\n"
        "EXECUTION MODEL (context only; do NOT include execution params in the JSON):\n"
        "- The Runner generates N samples by walking transformation sequences per dimension and decomposing the composite "
        "  tree (see Runner::gen and FeatureBase::transform). Seeds and per-node RunningInstances (startPoint, transformShift) "
        "  select which parts of each transformation stream are sampled.\n"
        "- Given the same feature definition and the same RunningInstances tree, execution is deterministic.\n"
        "\n"
        "OUTPUT REQUIREMENT:\n"
        "- Return ONLY the PT feature JSON object as specified above.\n"
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"OpenAI returned invalid JSON: {raw}") from e

    # Make name unique
    data["name"] = data.get("name", "feature") + str(int(time.time()))

    return data

def get_feature(base_url: str, access_token: str, refresh_token: str, name: str, version: str = None, timeout: float = 10.0) -> dict:
    """
    Calls GET /feature/{name}/{optional_version}
    Returns parsed JSON with feature definition.
    """
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "X-Refresh-Token": refresh_token,
    }

    if version:
        url = f"{base_url}/feature/{name}/{version}"
    else:
        url = f"{base_url}/feature/{name}"

    r = requests.get(url, headers=headers, timeout=timeout)
    return _handle_response(r)


def main():
    # Load or create account
    acct = _get_account()

    # 1) Nonce
    nonce = get_nonce(API_BASE, acct.address)
    print(f"Nonce: {nonce}")

    # 2) Sign message (personal_sign / EIP-191)
    message_text = f"Login nonce: {nonce}"
    signed = acct.sign_message(encode_defunct(text=message_text))
    signature = signed.signature.hex()

    # 3) /auth → expect access_token + refresh_token
    auth_res = post_auth(API_BASE, acct.address, message_text, signature)
    print("Auth response:")
    print(json.dumps(auth_res, indent=2))
    access_token = auth_res.get("access_token")
    refresh_token = auth_res.get("refresh_token")
    if not access_token or not refresh_token:
        print("Did not receive both access_token and refresh_token from /auth.", file=sys.stderr)
        sys.exit(1)

    time.sleep(1)

    # 4) /refresh using provided headers; prints new tokens
    refreshed = post_refresh(API_BASE, access_token, refresh_token)
    print("\nRefresh response:")
    print(json.dumps(refreshed, indent=2))
    new_access = refreshed.get("access_token")
    new_refresh = refreshed.get("refresh_token")
    if not (new_access and new_refresh):
        print("Refresh did not return both tokens.", file=sys.stderr)
        sys.exit(1)

    # ----------------------------------------------------------------
    # Phase 1: Vocabulary Builder (3 iterations for now)
    # ----------------------------------------------------------------
    vocabulary = []

    def pad_dimension_ops(d, min_len, default_op):
        """Ensure each dimension has at least min_len ops by appending defaults."""
        ops = d.get("transformations", [])
        while len(ops) < min_len:
            ops.append(default_op.copy())
        d["transformations"] = ops

    for i in range(NUM_VOCAB):
        user_prompt = (
            "TASK\n"
            "You will generate ONE vocabulary element for a Performative Transactions system.\n"
            "A vocabulary element is a PT that, when executed, yields ONE BAR of music.\n"
            "Return ONLY the PT JSON (no prose). Do NOT include explanations.\n\n"

            "GLOBAL RULES\n"
            "- Use ONLY transformations in {add, subtract}, each with exactly one unsigned integer arg.\n"
            "- Prefer small args (0–127). Avoid underflow.\n"
            "- Names are lowercase; feature_name must be resolvable ('pitch','time','duration','velocity','numerator','denominator').\n\n"
            "– PIANO RANGE: Design pitch so that, for any starting seed in [48..72], sampling up to 64 steps stays within MIDI [43..96] (C2–C7).\n"
            "– SEED ASSUMPTION: Composer will seed pitch in [48..72]; design cycles to remain within [43..96] for up to 64 steps from any seed in that range."

            "TEXTURE (CHOOSE ONE, THEN ADHERE STRICTLY)\n"
            "- Silently choose EXACTLY ONE texture mode (do not output which), then generate the PT to match it tightly.\n"
            "  A) MELODY ONLY\n"
            "     • Monophonic: no chord stacks.\n"
            "     • time: NO 'add 0' anywhere; strictly advancing with small positives.\n"
            "  B) CHORDS ONLY (block chords)\n"
            "     • Homorhythmic chord fields; little/no melodic passing.\n"
            "     • time: frequent chord bursts via runs of 'add 0' (50–90% of ops), with occasional small forward steps.\n"
            "  C) MELODY + OCCASIONAL CHORDS\n"
            "     • Primarily monophonic line with 1–2 brief chord bursts.\n"
            "     • time: 1–2 short runs of 'add 0' (each 1–3 ops), otherwise strictly advancing.\n"
            "  D) SINGLE SUSTAINED CHORD\n"
            "     • One harmony sustained across the bar; micro-variation allowed via pitch holds.\n"
            "     • time: ALL (or all but one) ops are 'add 0'; if needed, include exactly one small positive 'add' to ensure net advance > 0.\n"
            "  E) PROGRESSION OF CHORDS (no melody)\n"
            "     • Sequence of distinct chord sonorities; no linear melody.\n"
            "     • time: pattern of short forward steps separating brief 'add 0' bursts at each chord.\n"
            "- Across generations, vary texture choices; avoid repeating the same texture in successive elements if possible.\n\n"


            "METER (FROZEN)\n"
            "- For 'numerator' and 'denominator' use EXACTLY one transformation: {\"name\":\"add\",\"args\":[0]}.\n\n"

            "TIME (MONOTONIC; TEXTURE-DRIVEN CHORD BURSTS)\n"
            "- Use ONLY 'add' operations. No 'subtract'.\n"
            "- Overall, time must advance: the net sum of time args across the cycle is > 0.\n"
            "- Build a non-random pulse cell (len 3–5) and repeat it with one small variation. Total 4–12 ops.\n"
            "- CHORD BURSTS are permitted ONLY if the selected TEXTURE allows them; when allowed, implement them as runs of 'add 0'\n"
            "  (each burst 1–3 consecutive 'add 0'); otherwise, prohibit 'add 0' in time.\n"
            "- Outside permitted bursts, use strictly positive 'add' values (e.g., 1–3) to keep moving forward.\n\n"

            "PITCH (PIANO-SAFE COHERENCE + CIRCULARITY)\n"
            "- Use 8–16 transformations.\n"
            "- PIANO RANGE GUARANTEE: When sampled for up to 64 steps with ANY starting seed in [56..72], all resulting pitches must remain within [46..96] (C2–C7).\n"
            "- NET DRIFT CONTROL: Make the net sum over one cycle near zero (target in [−4..+4]) so long runs do not drift out of range.\n"
            "- INTERVAL VARIETY: Include AT LEAST TWO larger intervals from {5,7,12} (P4/P5/octave) at musically meaningful points, BUT pair them with corrective moves so the range guarantee holds (e.g., later subtract 5/7/12 or distribute smaller opposite steps).\n"
            "- HOLDS: You MAY use 'add 0' to sustain chord tones while time advances.\n"
            "- LOCAL STEP LIMITS: No more than 3 consecutive steps exclusively from {+1,+2} without a corrective move; avoid more than 3 consecutive 'add' with arg ≥5 in the same direction without an intervening 'subtract'.\n"
            "- CIRCULARITY: Use periodic returns, mirrored cells, rise-then-corrective fall, or cycling motifs that realign to maintain boundedness.\n"
            "- During any chord burst (where time uses 'add 0'), outline chord tones (e.g., 3,4,5,7,12) while still respecting the range guarantee.\n"

            "DURATION (QUANTIZED VALUES + CIRCULARITY)\n"
            "- Use 4–12 transformations.\n"
            "- Aim for a SMALL SET of distinct duration states (typically 2–3), akin to common note values\n"
            "  (e.g., quarter, eighth, sixteenth). Implement via many 'add 0' plateaus with occasional small toggles.\n"
            "- Constrain each arg to a small range (0–3), and avoid long monotonic runs; changes should be reversible.\n"
            "- CIRCULARITY: design the cycle to be near zero-sum so durations do not drift larger/smaller over long sampling.\n"
            "- Prefer keeping values stable on strong pulses and introducing brief shorter values around syncopations.\n\n"

            "VELOCITY (MIDI-SAFE CONTOUR + CIRCULARITY)\n"
            "- Use 3–8 transformations.\n"
            "- MIDI-safe: each arg must be small (0–8). Absolutely no double-digit jumps.\n"
            "- Prefer arcs (crescendo/decrescendo) or terraced contours with modest steps.\n"
            "- You MAY use 'add 0' to hold levels (plateaus).\n"
            "- Avoid more than 3 consecutive ops in the same direction; insert small corrective moves.\n"
            "- CIRCULARITY: design cycles so the net drift over a loop is near zero, ensuring seeds in [0,127] remain bounded even if sampled long.\n\n"

            "BAR COMPLETENESS\n"
            "- Include scalar dimensions for: pitch, time, duration, velocity, numerator, denominator.\n"
            "- Each dimension MUST have at least one transformation (respect counts above for richness).\n"
            "- Internal cycles should feel like one coherent bar under the (frozen) meter, and also remain coherent if sampled longer.\n\n"

            "DIVERSITY (AGAINST EXISTING_VOCAB)\n"
            "- Produce an element not near-duplicate of any existing one.\n"
            "- Ensure novelty by changing at least TWO of: (op,arg) multiset in a non-meter dimension; op count/length; add/subtract balance.\n\n"
            "- the chosen TEXTURE mode (avoid repeating the same mode as the most recent element in EXISTING_VOCAB).\n"


            "OUTPUT\n"
            "- Return ONLY the PT JSON with keys: {name, dimensions, feature_name, transformations, args}.\n\n"
            "EXISTING_VOCAB (do not output this section)\n"
            f"{json.dumps(vocabulary, ensure_ascii=False)}"
        )

        # Generate one PT
        feature_payload = generate_feature(user_prompt)

        # --- Safety nets / enforcement ---
        try:
            dims = feature_payload.get("dimensions", [])

            # Index by feature_name (lowercased) for quick edits
            by_name = { (d.get("feature_name") or "").strip().lower(): d for d in dims }

            # 1) Freeze meter
            for meter_name in ("numerator", "denominator"):
                d = by_name.get(meter_name)
                if d is not None:
                    d["transformations"] = [{"name": "add", "args": [0]}]

        except Exception:
            # let server-side validation speak if something unexpected happens
            pass

        print(f"\nGenerated vocabulary element {i+1}:")
        print(json.dumps(feature_payload, indent=2))

        # Post to DCN
        feature_res = post_feature(API_BASE, new_access, new_refresh, feature_payload)
        print(f"\nPOST /feature response for element {i+1}:")
        print(json.dumps(feature_res, indent=2))

        # Append to vocabulary so next iteration sees it
        vocabulary.append(feature_payload)

    # Save vocabulary locally for inspection
    with open("vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
        print(f"\nVocabulary of {len(vocabulary)} elements saved to vocabulary.json")

    # Enforce EXACT size
    if len(vocabulary) != NUM_VOCAB:
        print(f"Error: expected exactly NUM_VOCAB={NUM_VOCAB} elements, got {len(vocabulary)}.", file=sys.stderr)
        sys.exit(1)
    
    # ----------------------------------------------------------------
    # Phase 2: simplest composer
    # - execute each of the PTs in the vocabulary three times (vary N only for now)
    # - ask LLM to schedule them by assigning absolute start times
    # - apply schedule by shifting ONLY the 'time' streams
    # ----------------------------------------------------------------
    # Requires the DCN SDK (pip install dcn or your local module)
    import dcn

    # 2.1) Init SDK client and login with the SAME acct
    sdk_client = dcn.Client()
    sdk_client.login_with_account(acct)

    # 2.2) Helper: execute a PT and return a normalized dict
    def execute_pt(pt_name: str, N: int, seed_overrides: dict | None = None, transform_shifts: dict | None = None, dims: list | None = None):
        """
        Executes a PT with N samples and explicit RunningInstances.
        'dims' must be the feature's dimensions list (from the Phase 1 JSON).
        """
        if dims is None or not isinstance(dims, list):
            raise ValueError("execute_pt: 'dims' (feature dimensions list) is required")

        # sensible defaults
        seeds = {
            "pitch": random.randint(30, 80),
            "time": 0,
            "duration": random.choice([1, 2, 4, 8, 16]),
            "velocity": random.randint(30, 80),
            "numerator": 4,
            "denominator": 4,
        }
        if seed_overrides:
            seeds.update(seed_overrides)

        # RunningInstances size = root (1) + one per dimension, IN THE SAME ORDER as 'dims'
        running = [(0, 0)] * (1 + len(dims))
        running[0] = (0, 0)  # root

        for i, d in enumerate(dims):
            fname = (d.get("feature_name") or "").strip().lower()
            sp = int(seeds.get(fname, 0))
            ts = int((transform_shifts or {}).get(fname, 0))
            running[i + 1] = (sp, ts)

        # Execute
        result = sdk_client.execute(pt_name, N, running)

        # Normalize
        out = {"pt_name": pt_name, "N": N, "samples": result, "streams": {}}
        streams = {}
        for s in result:
            try:
                fp = getattr(s, "feature_path")
                data = getattr(s, "data")
            except Exception:
                fp = (s.get("feature_path") if isinstance(s, dict) else "") or ""
                data = (s.get("data") if isinstance(s, dict) else []) or []
            tail = (fp.split("/")[-1]).strip().lower() if fp else ""
            streams[tail] = list(data)
        out["streams"] = streams
        return out

    # 2.3) Plan: episodic form with variable takes and reprises
    TAKES_MIN, TAKES_MAX = 1, 5           # per-PT, per-episode
    EPISODES_MIN, EPISODES_MAX = 3, 6     # total episodes (not counting reprises)
    REPRISE_EPISODES_MIN, REPRISE_EPISODES_MAX = 1, 3   # extra episodes that revisit earlier PTs

    exec_plan = []
    episode_id = 0

    def add_episode(selected_pts, episode_id):
        """Append an episode: each selected PT gets a random number of takes."""
        for v in selected_pts:
            pt_name = v["name"]
            dims = v.get("dimensions", [])
            takes = random.randint(TAKES_MIN, TAKES_MAX)
            for _ in range(takes):
                run_id = len(exec_plan)
                N = random.randint(3, 25)  # short–long variety
                seed = {
                    "pitch": random.randint(30, 80),
                    "time": 0,
                    "duration": random.choice([1, 2, 4, 8, 16]),
                    "velocity": random.randint(30, 80),
                    "numerator": 4,
                    "denominator": 4,
                }
                exec_plan.append({
                    "run_id": run_id,
                    "episode": episode_id,
                    "pt_name": pt_name,
                    "N": N,
                    "seed": seed,
                    "dims": dims,
                })

    # --- Primary episodes (introduce material) ---
    num_episodes = random.randint(EPISODES_MIN, min(EPISODES_MAX, len(vocabulary)))
    shuffled_vocab = vocabulary[:]  # don’t mutate original
    random.shuffle(shuffled_vocab)

    # size of each episode = 2..min(5, NUM_VOCAB)
    cursor = 0
    for e in range(num_episodes):
        episode_size = random.randint(2, min(5, len(vocabulary)))
        # take slice from shuffled list; wrap if needed
        selected = []
        while len(selected) < episode_size:
            if cursor >= len(shuffled_vocab):
                cursor = 0
                random.shuffle(shuffled_vocab)  # fresh order if we wrapped
            selected.append(shuffled_vocab[cursor])
            cursor += 1
        add_episode(selected, episode_id)
        episode_id += 1

    # --- Reprise episodes (recall earlier material) ---
    num_reprises = random.randint(REPRISE_EPISODES_MIN, REPRISE_EPISODES_MAX)
    for _ in range(num_reprises):
        # pick 1–3 PTs from earlier episodes to reprise
        reprise_count = random.randint(1, min(3, len(vocabulary)))
        reprise_pts = random.sample(vocabulary, reprise_count)
        add_episode(reprise_pts, episode_id)
        episode_id += 1


    # 2.4) Execute with seeds
    runs = []
    for job in exec_plan:
        r = execute_pt(job["pt_name"], job["N"], seed_overrides=job["seed"], dims=job["dims"])
        r["run_id"] = job["run_id"]
        runs.append(r)


    # 2.5) Ask OpenAI to schedule: output only JSON with placements [{run_id, start_time}]
    #     We only allow shifting 'time' by adding a nonnegative integer offset.
    #     Provide simple stats per run to help the model avoid overlaps.
    client = OpenAI(api_key=OPENAI_API_KEY)

    def summarize_runs_for_prompt(runs_list):
        summary = []
        for r in runs_list:
            time_stream = r["streams"].get("time", [])
            tmin = int(min(time_stream)) if time_stream else 0
            tmax = int(max(time_stream)) if time_stream else 0
            tspan = tmax - tmin if time_stream else 0
            summary.append({
                "run_id": r["run_id"],
                "pt_name": r["pt_name"],
                "N": r["N"],
                "time_min": tmin,
                "time_max": tmax,
                "time_span": tspan,
                # You can include texture hints later if you store them in Phase 1
            })
        return summary

    schedule_system = (
        "ROLE:\n"
        "You are a composition scheduler. Output ONLY valid JSON with placements for a set of PT runs.\n"
        "You will receive a list of runs with their local time spans. Your job is to assign each run a non-negative\n"
        "absolute start time (integer) so the full piece forms one timeline. You MUST NOT change any data except\n"
        "by shifting time: for each run you specify 'start_time', and the composer will add that to every local time value.\n"
        "\n"
        "CORE AESTHETIC: BREATH & SPACING\n"
        "- Do NOT place runs back-to-back by default. Intentionally insert rests (silences) between many runs.\n"
        "- Target overall silence ratio ≈ 15–35% of the final timeline (rough guideline; keep rests musical, not huge).\n"
        "- You may also leave an initial pre-roll silence before the first run (e.g., a few units) and short codas after dense sections.\n"
        "\n"
        "GAP RULES (rests between runs)\n"
        "- Preferred gap length between adjacent runs: 4–15 time units. Occasionally allow a longer gap 13–20, but at most two of those.\n"
        "- Maximum single gap: 24. Avoid long dead zones.\n"
        "- Ensure at least ⌈(R−1)/2⌉ gaps are non-zero (R = number of runs). In other words: at least half the boundaries should have a gap.\n"
        "- Small overlaps are allowed (for dovetailing) but keep them modest: overlap ≤ 40% of the shorter run's span.\n"
        "\n"
        "ORDER & SCALE\n"
        "- Preserve the provided run order unless a small local swap (adjacent runs only) clearly improves spacing.\n"
        "- Keep offsets small and practical; you do not know the tempo.\n"
        "\n"
        "CONSTRAINTS (OUTPUT SHAPE)\n"
        "- Output exactly this JSON shape:\n"
        "{\n"
        '  \"placements\": [ {\"run_id\": <int>, \"start_time\": <int>}, ... ]\n'
        "}\n"
        "- 'start_time' must be >= 0. No extra keys. No comments. No prose.\n"
    )

    schedule_user = (
        "Below are the runs to schedule. For each, you get run_id and (time_min, time_max, time_span) in LOCAL units.\n"
        "Choose a sensible 'start_time' for each run so they form one longer composition with intentional SILENCE between many runs.\n"
        "Aim for an overall silence ratio around 15–35%, using gaps mostly 3–12 units, with at most two longer gaps (13–20). Max any gap 24.\n"
        "Keep overlaps modest (≤40% of the shorter span) and preserve order unless a small adjacent swap improves spacing.\n"
        "Remember: the composer will ONLY add 'start_time' to each local 'time' value. Do not attempt any other edits.\n\n"
        f"{json.dumps(summarize_runs_for_prompt(runs), ensure_ascii=False)}"
    )


    sched_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": schedule_system},
            {"role": "user", "content": schedule_user},
        ],
        temperature=0.5,
    )
    sched_raw = sched_resp.choices[0].message.content.strip()
    try:
        schedule = json.loads(sched_raw)
        placements = schedule.get("placements", [])
    except json.JSONDecodeError as e:
        raise ValueError(f"Scheduler returned invalid JSON: {sched_raw}") from e

    # 2.6) Build a map run_id -> start_time
    start_map = {int(p["run_id"]): int(p["start_time"]) for p in placements if "run_id" in p and "start_time" in p}

    # 2.7) Apply schedule: add 'start_time' to ONLY the 'time' stream of each run
    adjusted_runs = []
    for r in runs:
        run_id = r["run_id"]
        start = int(start_map.get(run_id, 0))
        # shift only the 'time' sample arrays in-place and regenerate 'samples' with updated time entries
        streams = r["streams"]
        time_stream = streams.get("time", [])
        shifted_time = [int(t) + start for t in time_stream]

        # Rebuild samples array: update entries whose feature_path tail == 'time'
        new_samples = []
        new_samples = []
        for s in r["samples"]:
            try:
                fp = getattr(s, "feature_path")
                data = getattr(s, "data")
            except Exception:
                fp = (s.get("feature_path") if isinstance(s, dict) else "") or ""
                data = (s.get("data") if isinstance(s, dict) else []) or []

            tail = (fp.split("/")[-1]).strip().lower() if fp else ""
            if tail == "time":
                new_samples.append({"feature_path": fp, "data": shifted_time})
            else:
                # keep original as plain dict to make JSON-serializable
                new_samples.append({"feature_path": fp, "data": list(data)})


        adjusted_runs.append({
            "run_id": run_id,
            "pt_name": r["pt_name"],
            "N": r["N"],
            "start_time": start,
            "samples": new_samples,   # time-shifted
        })

    # 2.7c) Build a single merged timeline for the MIDI player.
    # Turn each run into events (time-aligned tuples), merge, sort by time, then split back into six arrays.

    REQUIRED_STREAMS = ["pitch", "time", "duration", "velocity", "numerator", "denominator"]

    def clamp01_127(x): 
        return max(0, min(127, int(x)))

    all_events = []  # list of dicts: {time, pitch, duration, velocity, numerator, denominator, source_pt, source_run}

    for r in adjusted_runs:
        # Extract streams for this run as lists
        # Each stream length should be N; if any is missing or lengths mismatch, we skip this run safely.
        streams = {}
        for s in r["samples"]:
            # support both dicts and SDK objects
            try:
                fp = getattr(s, "feature_path")
                data = getattr(s, "data")
            except Exception:
                fp = (s.get("feature_path") if isinstance(s, dict) else "") or ""
                data = (s.get("data") if isinstance(s, dict) else []) or []
            tail = (fp.split("/")[-1]).strip().lower() if fp else ""
            streams[tail] = list(data)

        # Validate presence
        if not all(k in streams for k in REQUIRED_STREAMS):
            # If something's missing, skip this run (or you could fill sensible defaults)
            continue

        # Validate equal lengths
        lengths = [len(streams[k]) for k in REQUIRED_STREAMS]
        if len(set(lengths)) != 1:
            # If lengths differ, truncate to the shortest to preserve alignment
            L = min(lengths)
        else:
            L = lengths[0]

        # Clamp MIDI-ish ranges, ensure ints
        pit = [clamp01_127(v) for v in streams["pitch"][:L]]
        tim = [int(v) for v in streams["time"][:L]]
        dur = [max(0, int(v)) for v in streams["duration"][:L]]
        vel = [clamp01_127(v) for v in streams["velocity"][:L]]
        # Force 4/4 meter for the player payload
        num = [4] * L
        den = [4] * L

        for i in range(L):
            all_events.append({
                "time": tim[i],
                "pitch": pit[i],
                "duration": dur[i],
                "velocity": vel[i],
                "numerator": num[i],
                "denominator": den[i],
                "source_pt": r["pt_name"],
                "source_run": r["run_id"],
            })

    # Sort globally by time; stable sort keeps source order for identical times
    all_events.sort(key=lambda e: e["time"])

    # Split back into single concatenated arrays
    concat_pitch       = [e["pitch"]       for e in all_events]
    concat_time        = [e["time"]        for e in all_events]
    concat_duration    = [e["duration"]    for e in all_events]
    concat_velocity    = [e["velocity"]    for e in all_events]
    concat_numerator   = [e["numerator"]   for e in all_events]
    concat_denominator = [e["denominator"] for e in all_events]

    # Emit exactly the shape your MIDI player expects (one entry per scalar stream)
    player_payload_merged = [
        {"feature_path": "/composition/pitch",       "data": concat_pitch},
        {"feature_path": "/composition/time",        "data": concat_time},
        {"feature_path": "/composition/duration",    "data": concat_duration},
        {"feature_path": "/composition/velocity",    "data": concat_velocity},
        {"feature_path": "/composition/numerator",   "data": concat_numerator},
        {"feature_path": "/composition/denominator", "data": concat_denominator},
    ]

    with open("player_payload_merged.json", "w", encoding="utf-8") as f:
        json.dump(player_payload_merged, f, ensure_ascii=False, indent=2)

    print("\n=== COPY THIS INTO YOUR MIDI PLAYER (MERGED) ===")
    print(json.dumps(player_payload_merged, ensure_ascii=False, indent=2))
    print("=== END PLAYER PAYLOAD (MERGED) ===\n")


    # 2.8) Save the composed result (all time-shifted runs)
    with open("phase2_composition.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocabulary_names_used": [v["name"] for v in vocabulary],  # exactly NUM_VOCAB
            "placements": placements,
            "runs_time_shifted": adjusted_runs
        }, f, ensure_ascii=False, indent=2)

    print("\nPhase 2 complete. Wrote time-aligned composition to phase2_composition.json")

if __name__ == "__main__":
    try:
        NUM_VOCAB = int(sys.argv[1])
    except Exception:
        NUM_VOCAB = DEFAULT_NUM_VOCAB
    main()