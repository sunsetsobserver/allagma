#!/usr/bin/env python3
import argparse
import copy
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time

import requests
from eth_account import Account
from eth_account.messages import encode_defunct
from openai import OpenAI

try:
    from secrets import OPENAI_API_KEY as _OPENAI_API_KEY
except Exception:
    _OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

API_BASE = os.getenv("API_BASE", "https://api.decentralised.art")
DEFAULT_NUM_VOCAB = 8

SCALAR_DIM_ORDER = ("time", "duration", "pitch", "velocity", "numerator", "denominator")
REQUIRED_STREAMS = ("pitch", "time", "duration", "velocity", "numerator", "denominator")


def _require_openai_key() -> str:
    if not _OPENAI_API_KEY:
        raise RuntimeError(
            "Missing OpenAI API key. Set OPENAI_API_KEY env var or provide secrets.py with OPENAI_API_KEY."
        )
    return _OPENAI_API_KEY


def _sanitize_solidity_identifier(name: str, *, prefix: str, max_len: int = 96) -> str:
    """
    Solidity contract identifiers must match [A-Za-z_][A-Za-z0-9_]*.
    """
    raw = (name or "").strip()
    if not raw:
        raw = prefix

    out = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        out = prefix
    if not (out[0].isalpha() or out[0] == "_"):
        out = f"{prefix}_{out}"

    if len(out) > max_len:
        # Keep suffix entropy to avoid collisions when truncating.
        tail = out[-12:]
        out = f"{out[:max_len - 13]}_{tail}"
    return out


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _maybe_export_midi(workdir: pathlib.Path):
    """
    Optional auto-export: convert player_payload_merged.json -> player_payload_merged.mid
    using Node tool tools/pt2midi.js.
    """
    if _env_bool("NO_MIDI", False):
        print("MIDI export skipped (NO_MIDI=1).")
        return

    tools_js = workdir / "tools" / "pt2midi.js"
    input_json = workdir / "player_payload_merged.json"
    output_mid = workdir / "player_payload_merged.mid"

    if not tools_js.exists():
        print("MIDI export skipped (tools/pt2midi.js not found).")
        return
    if not input_json.exists():
        print("MIDI export skipped (player_payload_merged.json not found).")
        return

    try:
        result = subprocess.run(
            ["node", str(tools_js), str(input_json), str(output_mid)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if output_mid.exists():
            print(f"Wrote MIDI: {output_mid}")
        else:
            print("MIDI export completed, but output file was not found.")
    except FileNotFoundError:
        print("MIDI export skipped (Node not found on PATH).")
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "").strip()
        print("MIDI export failed (non-fatal).")
        if "Cannot find module 'jzz'" in output:
            print("Install MIDI deps once: npm install")
        if output:
            print(output)
    except Exception as exc:
        print(f"MIDI export failed (non-fatal): {exc}")


class DCNClient:
    REQUIRED_TRANSFORMATIONS = {
        "add": "return x + args[0];",
        "subtract": "return x - args[0];",
    }

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self.access_token = None

    def _handle_response(self, r: requests.Response):
        try:
            data = r.json()
        except json.JSONDecodeError:
            r.raise_for_status()
            return {"raw": r.text}
        if not r.ok:
            print(f" fail: {r.status_code} {data}", file=sys.stderr)
            raise requests.HTTPError(f" fail: {r.status_code} {data}", response=r)
        return data

    def _authz_headers(self):
        if not self.access_token:
            return {}
        return {"Authorization": f"Bearer {self.access_token}"}

    def get_nonce(self, address: str) -> str:
        url = f"{self.base_url}/nonce/{address}"
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "nonce" in data:
            return str(data["nonce"])
        raise ValueError(f"Unexpected nonce response shape: {data}")

    def post_auth(self, address: str, message: str, signature: str):
        r = self.session.post(
            f"{self.base_url}/auth",
            json={"address": address, "message": message, "signature": signature},
            timeout=self.timeout,
        )
        data = self._handle_response(r)
        self.access_token = data.get("access_token")
        return data

    def ensure_auth(self, acct: Account):
        if self.access_token:
            return
        nonce = self.get_nonce(acct.address)
        message = f"Login nonce: {nonce}"
        signature = acct.sign_message(encode_defunct(text=message)).signature.hex()
        auth_res = self.post_auth(acct.address, message, signature)
        if not self.access_token:
            raise RuntimeError(f"Auth failed — missing access token: {auth_res}")

    def _post_with_reauth(self, path: str, payload: dict, acct: Account):
        self.ensure_auth(acct)
        url = f"{self.base_url}{path}"
        r = self.session.post(url, json=payload, headers=self._authz_headers(), timeout=self.timeout)
        if r.status_code == 401:
            self.access_token = None
            self.ensure_auth(acct)
            r = self.session.post(url, json=payload, headers=self._authz_headers(), timeout=self.timeout)
        return r

    def post_feature(self, payload: dict, acct: Account):
        return self._handle_response(self._post_with_reauth("/feature", payload, acct))

    def post_particle(self, payload: dict, acct: Account):
        return self._handle_response(self._post_with_reauth("/particle", payload, acct))

    def execute_particle(self, payload: dict, acct: Account):
        data = self._handle_response(self._post_with_reauth("/execute", payload, acct))
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected /execute response shape: {type(data).__name__}")
        return data

    def has_transformation(self, name: str) -> bool:
        r = self.session.get(
            f"{self.base_url}/transformation/{name}",
            headers=self._authz_headers(),
            timeout=self.timeout,
        )
        if r.status_code == 404:
            return False
        if r.ok:
            return True
        body = (r.text or "").strip().replace("\n", " ")
        raise RuntimeError(f"Failed to check transformation '{name}': {r.status_code} {body}")

    def post_transformation(self, payload: dict, acct: Account):
        return self._handle_response(self._post_with_reauth("/transformation", payload, acct))

    def ensure_required_transformations(self, acct: Account):
        self.ensure_auth(acct)
        for name, sol_src in self.REQUIRED_TRANSFORMATIONS.items():
            if self.has_transformation(name):
                continue
            self.post_transformation({"name": name, "sol_src": sol_src}, acct)
            if not self.has_transformation(name):
                raise RuntimeError(f"Transformation '{name}' could not be created.")

    def preflight_endpoints(self, acct: Account):
        checks = [
            (
                "/feature",
                {"_preflight": True},
                {"name": "my_feature", "dimensions": [{"transformations": [{"name": "add", "args": [1]}]}]},
            ),
            (
                "/particle",
                {"_preflight": True},
                {
                    "name": "my_particle",
                    "feature_name": "my_feature",
                    "composite_names": ["", "", "", "", "", ""],
                    "condition_name": "",
                    "condition_args": [],
                },
            ),
            (
                "/execute",
                {"_preflight": True},
                {
                    "particle_name": "my_particle",
                    "samples_count": 4,
                    "running_instances": [{"start_point": 0, "transformation_shift": 0}],
                },
            ),
        ]
        for path, invalid_payload, sample_payload in checks:
            r = self._post_with_reauth(path, invalid_payload, acct)
            if r.status_code in (200, 201, 204, 400):
                continue
            body = (r.text or "").strip().replace("\n", " ")
            raise RuntimeError(
                f"Preflight failed on {path}: status={r.status_code}, response={body}, "
                f"sample_payload={json.dumps(sample_payload)}"
            )


def _get_account(private_key: str | None) -> Account:
    priv = private_key or os.getenv("PRIVATE_KEY")
    if priv:
        acct = Account.from_key(priv)
        print("Loaded account from PRIVATE_KEY/--private-key.")
    else:
        acct = Account.create("KEYSMASH FJAFJKLDSKF7JKFDJ 1530")
        print("Created example local account (ephemeral).")
    print(f"Address: {acct.address}")
    return acct


def generate_feature(user_prompt: str, model: str) -> dict:
    client = OpenAI(api_key=_require_openai_key())
    system_msg = (
        "ROLE:\n"
        "You output ONLY a single valid JSON object describing a DCN Performative Transaction (PT) FEATURE DEFINITION. "
        "Never include prose, code fences, or comments.\n"
        "\n"
        "INTENT:\n"
        "The JSON you return will be used by an off-chain adapter that deploys a Feature and wraps it in a Particle for execution.\n"
        "\n"
        "REQUIRED JSON SHAPE:\n"
        "{\n"
        "  \"name\": \"<string>\",\n"
        "  \"dimensions\": [\n"
        "    {\n"
        "      \"feature_name\": \"<string>\",\n"
        "      \"transformations\": [\n"
        "        {\"name\": \"<string>\", \"args\": [<uint32>]}\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "\n"
        "HARD CONSTRAINTS:\n"
        "- Output EXACTLY one JSON object; allowed keys ONLY: {name, dimensions, feature_name, transformations, args}.\n"
        "- ALLOWED transformation names ONLY: [\"add\", \"subtract\"].\n"
        "- For every transformation, 'args' MUST be an array with EXACTLY ONE unsigned integer.\n"
        "- Keep JSON compact and strictly valid.\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"OpenAI returned invalid JSON: {raw}") from e

    base_name = f"{data.get('name', 'feature')}_{int(time.time() * 1000)}_{os.urandom(2).hex()}"
    data["name"] = _sanitize_solidity_identifier(base_name, prefix="feature")
    return data


def _canonicalize_feature(feature_payload: dict) -> dict:
    dims_in = list(feature_payload.get("dimensions") or [])
    by_name = {}
    for dim in dims_in:
        fname = str(dim.get("feature_name") or "").strip().lower()
        if not fname:
            continue
        tr_out = []
        for tr in list(dim.get("transformations") or []):
            tr_name = str(tr.get("name") or "").strip()
            args = list(tr.get("args") or [])
            if not tr_name:
                continue
            if len(args) != 1:
                continue
            tr_out.append({"name": tr_name, "args": [int(args[0])]})
        if tr_out:
            by_name[fname] = {"feature_name": fname, "transformations": tr_out}

    missing = [name for name in SCALAR_DIM_ORDER if name not in by_name]
    if missing:
        raise RuntimeError(f"Generated feature missing required dimensions: {missing}")

    return {
        "name": str(feature_payload.get("name") or "").strip(),
        "dimensions": [by_name[name] for name in SCALAR_DIM_ORDER],
    }


def _feature_deploy_payload(feature_payload: dict) -> dict:
    return {
        "name": feature_payload["name"],
        "dimensions": [
            {
                "transformations": [
                    {
                        "name": str(tr["name"]),
                        "args": [int(tr["args"][0])],
                    }
                    for tr in dim["transformations"]
                ]
            }
            for dim in feature_payload["dimensions"]
        ],
    }


def _particle_payload_for_feature(feature_name: str, dimensions_count: int) -> dict:
    particle_name = _sanitize_solidity_identifier(f"{feature_name}__particle", prefix="particle")
    return {
        "name": particle_name,
        "feature_name": feature_name,
        "composite_names": [""] * int(dimensions_count),
        "condition_name": "",
        "condition_args": [],
    }


def _build_running_instances(seeds: dict, dims: list[dict], transform_shifts: dict | None = None) -> list[dict]:
    running = [{
        "start_point": int(seeds.get("time", 0)),
        "transformation_shift": 0,
    }]
    for dim in dims:
        fname = str(dim.get("feature_name") or "").strip().lower()
        running.append({
            "start_point": int(seeds.get(fname, 0)),
            "transformation_shift": int((transform_shifts or {}).get(fname, 0)),
        })
    return running


def _parse_dim_id_from_path(path: str) -> int | None:
    m = re.search(r":(\d+)$", path.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _normalize_execute_samples(samples: list[dict], dims: list[dict]) -> tuple[dict, list]:
    streams = {}
    unknown_paths = []
    ordered_names = [str(d.get("feature_name") or "").strip().lower() for d in dims]

    for sample in samples:
        path = str(sample.get("path") or sample.get("feature_path") or "").strip()
        data = [int(v) for v in list(sample.get("data") or [])]
        if not path:
            unknown_paths.append("<missing path>")
            continue

        dim_id = _parse_dim_id_from_path(path)
        if dim_id is not None and 0 <= dim_id < len(ordered_names):
            streams[ordered_names[dim_id]] = data
            continue

        tail = path.split("/")[-1].strip().lower()
        if tail in SCALAR_DIM_ORDER:
            streams[tail] = data
            continue

        unknown_paths.append(path)

    return streams, unknown_paths


def _streams_to_samples(streams: dict, pt_name: str) -> list[dict]:
    return [
        {"feature_path": f"/{pt_name}/{scalar}", "data": list(streams.get(scalar, []))}
        for scalar in SCALAR_DIM_ORDER
    ]


def _default_seeds() -> dict:
    return {
        "pitch": random.randint(30, 80),
        "time": 0,
        "duration": random.choice([1, 2, 4, 8, 16]),
        "velocity": random.randint(30, 80),
        "numerator": 4,
        "denominator": 4,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a composition using DCN feature+particle flow.")
    parser.add_argument("num_vocab", nargs="?", type=int, default=DEFAULT_NUM_VOCAB)
    parser.add_argument("--private-key", default=None, help="Hex EVM private key (overrides PRIVATE_KEY env).")
    parser.add_argument("--api-base", default=API_BASE, help="DCN API base URL.")
    parser.add_argument("--feature-model", default=os.getenv("FEATURE_MODEL", "gpt-4.1"))
    parser.add_argument("--scheduler-model", default=os.getenv("SCHEDULER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("DCN_TIMEOUT", "15")))
    args = parser.parse_args()
    workdir = pathlib.Path(__file__).resolve().parent

    num_vocab = int(args.num_vocab)
    if num_vocab <= 0:
        raise RuntimeError("num_vocab must be > 0")

    acct = _get_account(args.private_key)
    dcn = DCNClient(args.api_base, timeout=args.timeout)
    dcn.ensure_auth(acct)
    print("Auth response: access token received")

    print("Ensuring required transformations (add, subtract)...")
    dcn.ensure_required_transformations(acct)
    print("Running endpoint preflight checks (/feature, /particle, /execute)...")
    dcn.preflight_endpoints(acct)

    vocabulary = []
    particle_by_feature = {}

    for i in range(num_vocab):
        user_prompt = (
            "TASK\n"
            "You will generate ONE vocabulary element for a Performative Transactions system.\n"
            "A vocabulary element is a PT that, when executed, yields ONE BAR of music.\n"
            "Return ONLY the PT JSON (no prose). Do NOT include explanations.\n\n"
            "GLOBAL RULES\n"
            "- Use ONLY transformations in {add, subtract}, each with exactly one unsigned integer arg.\n"
            "- Prefer small args (0-127). Avoid underflow.\n"
            "- Names are lowercase; feature_name must be resolvable ('pitch','time','duration','velocity','numerator','denominator').\n\n"
            "- PIANO RANGE: Design pitch so that, for any starting seed in [48..72], sampling up to 64 steps stays within MIDI [36..96] (C2-C7).\n"
            "- SEED ASSUMPTION: Composer will seed pitch in [48..72]; design cycles to remain within [36..96] for up to 64 steps from any seed in that range.\n\n"
            "TEXTURE (CHOOSE ONE, THEN ADHERE STRICTLY)\n"
            "- Silently choose EXACTLY ONE texture mode (do not output which), then generate the PT to match it tightly.\n"
            "  A) MELODY ONLY\n"
            "     - Monophonic: no chord stacks.\n"
            "     - time: NO 'add 0' anywhere; strictly advancing with small positives.\n"
            "  B) CHORDS ONLY (block chords)\n"
            "     - Homorhythmic chord fields; little/no melodic passing.\n"
            "     - time: frequent chord bursts via runs of 'add 0' (50-90% of ops), with occasional small forward steps.\n"
            "  C) MELODY + OCCASIONAL CHORDS\n"
            "     - Primarily monophonic line with 1-2 brief chord bursts.\n"
            "     - time: 1-2 short runs of 'add 0' (each 1-3 ops), otherwise strictly advancing.\n"
            "  D) SINGLE SUSTAINED CHORD\n"
            "     - One harmony sustained across the bar; micro-variation allowed via pitch holds.\n"
            "     - time: ALL (or all but one) ops are 'add 0'; if needed, include exactly one small positive 'add' to ensure net advance > 0.\n"
            "  E) PROGRESSION OF CHORDS (no melody)\n"
            "     - Sequence of distinct chord sonorities; no linear melody.\n"
            "     - time: pattern of short forward steps separating brief 'add 0' bursts at each chord.\n"
            "- Across generations, vary texture choices; avoid repeating the same texture in successive elements if possible.\n\n"
            "METER (FROZEN)\n"
            "- For 'numerator' and 'denominator' use EXACTLY one transformation: {\"name\":\"add\",\"args\":[0]}.\n\n"
            "TIME (MONOTONIC; TEXTURE-DRIVEN CHORD BURSTS)\n"
            "- Use ONLY 'add' operations. No 'subtract'.\n"
            "- Overall, time must advance: the net sum of time args across the cycle is > 0.\n"
            "- Build a non-random pulse cell (len 3-5) and repeat it with one small variation. Total 4-12 ops.\n"
            "- CHORD BURSTS are permitted ONLY if the selected TEXTURE allows them; when allowed, implement them as runs of 'add 0'\n"
            "  (each burst 1-3 consecutive 'add 0'); otherwise, prohibit 'add 0' in time.\n"
            "- Outside permitted bursts, use strictly positive 'add' values (e.g., 1-3) to keep moving forward.\n\n"
            "PITCH (PIANO-SAFE COHERENCE + CIRCULARITY)\n"
            "- Use 8-16 transformations.\n"
            "- PIANO RANGE GUARANTEE: When sampled for up to 64 steps with ANY starting seed in [48..72], all resulting pitches must remain within [36..96] (C2-C7).\n"
            "- NET DRIFT CONTROL: Make the net sum over one cycle near zero (target in [-4..+4]) so long runs do not drift out of range.\n"
            "- INTERVAL VARIETY: Include AT LEAST TWO larger intervals from {5,7,12} at musically meaningful points.\n"
            "- HOLDS: You MAY use 'add 0' to sustain chord tones while time advances.\n"
            "- CIRCULARITY: Use periodic returns, mirrored cells, rise-then-corrective fall, or cycling motifs that realign to maintain boundedness.\n\n"
            "DURATION (QUANTIZED VALUES + CIRCULARITY)\n"
            "- Use 4-12 transformations.\n"
            "- Aim for a SMALL SET of distinct duration states (typically 2-3), akin to common note values.\n"
            "- Constrain each arg to a small range (0-3), and avoid long monotonic runs.\n\n"
            "VELOCITY (MIDI-SAFE CONTOUR + CIRCULARITY)\n"
            "- Use 3-8 transformations.\n"
            "- MIDI-safe: each arg must be small (0-8). Absolutely no double-digit jumps.\n\n"
            "BAR COMPLETENESS\n"
            "- Include scalar dimensions for: pitch, time, duration, velocity, numerator, denominator.\n"
            "- Each dimension MUST have at least one transformation.\n\n"
            "OUTPUT\n"
            "- Return ONLY the PT JSON with keys: {name, dimensions, feature_name, transformations, args}.\n\n"
            "EXISTING_VOCAB (do not output this section)\n"
            f"{json.dumps(vocabulary, ensure_ascii=False)}"
        )

        feature_payload = generate_feature(user_prompt, model=args.feature_model)
        feature_payload = _canonicalize_feature(feature_payload)

        # Freeze meter regardless of model output.
        for dim in feature_payload["dimensions"]:
            if dim["feature_name"] in ("numerator", "denominator"):
                dim["transformations"] = [{"name": "add", "args": [0]}]

        print(f"\nGenerated vocabulary element {i + 1}:")
        print(json.dumps(feature_payload, indent=2))

        deploy_payload = _feature_deploy_payload(feature_payload)
        feature_res = dcn.post_feature(deploy_payload, acct)
        print(f"\nPOST /feature response for element {i + 1}:")
        print(json.dumps(feature_res, indent=2))

        particle_payload = _particle_payload_for_feature(
            feature_name=feature_payload["name"],
            dimensions_count=len(feature_payload["dimensions"]),
        )
        particle_res = dcn.post_particle(particle_payload, acct)
        print(f"\nPOST /particle response for element {i + 1}:")
        print(json.dumps(particle_res, indent=2))

        vocabulary.append(feature_payload)
        particle_by_feature[feature_payload["name"]] = particle_payload["name"]

    with open("vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    print(f"\nVocabulary of {len(vocabulary)} elements saved to vocabulary.json")

    if len(vocabulary) != num_vocab:
        print(f"Error: expected exactly num_vocab={num_vocab} elements, got {len(vocabulary)}.", file=sys.stderr)
        sys.exit(1)

    def execute_pt(
        pt_name: str,
        particle_name: str,
        n_samples: int,
        dims: list[dict],
        seed_overrides: dict | None = None,
        transform_shifts: dict | None = None,
    ):
        seeds = _default_seeds()
        if seed_overrides:
            seeds.update(seed_overrides)

        running_instances = _build_running_instances(seeds, dims, transform_shifts)
        exec_payload = {
            "particle_name": particle_name,
            "samples_count": int(n_samples),
            "running_instances": running_instances,
        }
        samples = dcn.execute_particle(exec_payload, acct)
        streams, unknown = _normalize_execute_samples(samples, dims)

        missing = [name for name in SCALAR_DIM_ORDER if name not in streams]
        if missing:
            raise RuntimeError(
                f"Execute missing scalar streams for {pt_name}: {missing}. "
                f"Unknown paths: {unknown[:8]}"
            )

        return {
            "pt_name": pt_name,
            "particle_name": particle_name,
            "N": int(n_samples),
            "samples": samples,
            "streams": streams,
        }

    TAKES_MIN, TAKES_MAX = 1, 5
    EPISODES_MIN, EPISODES_MAX = 3, 6
    REPRISE_EPISODES_MIN, REPRISE_EPISODES_MAX = 1, 3

    exec_plan = []
    episode_id = 0

    def add_episode(selected_pts, episode):
        for feature in selected_pts:
            pt_name = feature["name"]
            particle_name = particle_by_feature[pt_name]
            dims = feature.get("dimensions", [])
            takes = random.randint(TAKES_MIN, TAKES_MAX)
            for _ in range(takes):
                run_id = len(exec_plan)
                n_samples = random.randint(3, 25)
                exec_plan.append({
                    "run_id": run_id,
                    "episode": episode,
                    "pt_name": pt_name,
                    "particle_name": particle_name,
                    "N": n_samples,
                    "seed": _default_seeds(),
                    "dims": dims,
                })

    num_episodes = random.randint(EPISODES_MIN, min(EPISODES_MAX, len(vocabulary)))
    shuffled_vocab = vocabulary[:]
    random.shuffle(shuffled_vocab)
    cursor = 0

    for _ in range(num_episodes):
        episode_size = random.randint(2, min(5, len(vocabulary)))
        selected = []
        while len(selected) < episode_size:
            if cursor >= len(shuffled_vocab):
                cursor = 0
                random.shuffle(shuffled_vocab)
            selected.append(shuffled_vocab[cursor])
            cursor += 1
        add_episode(selected, episode_id)
        episode_id += 1

    num_reprises = random.randint(REPRISE_EPISODES_MIN, REPRISE_EPISODES_MAX)
    for _ in range(num_reprises):
        reprise_count = random.randint(1, min(3, len(vocabulary)))
        reprise_pts = random.sample(vocabulary, reprise_count)
        add_episode(reprise_pts, episode_id)
        episode_id += 1

    runs = []
    for job in exec_plan:
        run = execute_pt(
            job["pt_name"],
            job["particle_name"],
            job["N"],
            dims=job["dims"],
            seed_overrides=job["seed"],
        )
        run["run_id"] = job["run_id"]
        runs.append(run)

    def summarize_runs_for_prompt(runs_list):
        summary = []
        for run in runs_list:
            time_stream = run["streams"].get("time", [])
            tmin = int(min(time_stream)) if time_stream else 0
            tmax = int(max(time_stream)) if time_stream else 0
            tspan = tmax - tmin if time_stream else 0
            summary.append({
                "run_id": run["run_id"],
                "pt_name": run["pt_name"],
                "N": run["N"],
                "time_min": tmin,
                "time_max": tmax,
                "time_span": tspan,
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
        "- Target overall silence ratio around 15-35% of the final timeline.\n"
        "- You may leave an initial pre-roll silence before the first run.\n"
        "\n"
        "GAP RULES\n"
        "- Preferred gap length between adjacent runs: 4-15 time units.\n"
        "- Maximum single gap: 24.\n"
        "- Ensure at least half the boundaries have a non-zero gap.\n"
        "- Small overlaps are allowed (<=40% of the shorter run's span).\n"
        "\n"
        "CONSTRAINTS (OUTPUT SHAPE)\n"
        "- Output exactly this JSON shape:\n"
        "{\n"
        '  "placements": [ {"run_id": <int>, "start_time": <int>}, ... ]\n'
        "}\n"
        "- 'start_time' must be >= 0. No extra keys. No comments. No prose.\n"
    )

    schedule_user = (
        "Below are the runs to schedule. For each, you get run_id and (time_min, time_max, time_span) in LOCAL units.\n"
        "Choose a sensible 'start_time' for each run so they form one longer composition with intentional SILENCE between many runs.\n"
        "Remember: the composer will ONLY add 'start_time' to each local 'time' value.\n\n"
        f"{json.dumps(summarize_runs_for_prompt(runs), ensure_ascii=False)}"
    )

    client = OpenAI(api_key=_require_openai_key())
    sched_resp = client.chat.completions.create(
        model=args.scheduler_model,
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

    start_map = {
        int(p["run_id"]): int(p["start_time"])
        for p in placements
        if "run_id" in p and "start_time" in p
    }

    adjusted_runs = []
    for run in runs:
        run_id = run["run_id"]
        start = int(start_map.get(run_id, 0))

        shifted_streams = copy.deepcopy(run["streams"])
        shifted_streams["time"] = [int(t) + start for t in shifted_streams.get("time", [])]
        shifted_samples = _streams_to_samples(shifted_streams, run["pt_name"])

        adjusted_runs.append({
            "run_id": run_id,
            "pt_name": run["pt_name"],
            "particle_name": run["particle_name"],
            "N": run["N"],
            "start_time": start,
            "samples": shifted_samples,
            "streams": shifted_streams,
        })

    def clamp01_127(x):
        return max(0, min(127, int(x)))

    all_events = []
    for run in adjusted_runs:
        streams = run["streams"]
        if not all(k in streams for k in REQUIRED_STREAMS):
            continue
        lengths = [len(streams[k]) for k in REQUIRED_STREAMS]
        sample_len = min(lengths)
        if sample_len <= 0:
            continue

        pit = [clamp01_127(v) for v in streams["pitch"][:sample_len]]
        tim = [int(v) for v in streams["time"][:sample_len]]
        dur = [max(0, int(v)) for v in streams["duration"][:sample_len]]
        vel = [clamp01_127(v) for v in streams["velocity"][:sample_len]]
        num = [4] * sample_len
        den = [4] * sample_len

        for i in range(sample_len):
            all_events.append({
                "time": tim[i],
                "pitch": pit[i],
                "duration": dur[i],
                "velocity": vel[i],
                "numerator": num[i],
                "denominator": den[i],
                "source_pt": run["pt_name"],
                "source_run": run["run_id"],
            })

    all_events.sort(key=lambda event: event["time"])

    concat_pitch = [e["pitch"] for e in all_events]
    concat_time = [e["time"] for e in all_events]
    concat_duration = [e["duration"] for e in all_events]
    concat_velocity = [e["velocity"] for e in all_events]
    concat_numerator = [e["numerator"] for e in all_events]
    concat_denominator = [e["denominator"] for e in all_events]

    player_payload_merged = [
        {"feature_path": "/composition/pitch", "data": concat_pitch},
        {"feature_path": "/composition/time", "data": concat_time},
        {"feature_path": "/composition/duration", "data": concat_duration},
        {"feature_path": "/composition/velocity", "data": concat_velocity},
        {"feature_path": "/composition/numerator", "data": concat_numerator},
        {"feature_path": "/composition/denominator", "data": concat_denominator},
    ]

    with open("player_payload_merged.json", "w", encoding="utf-8") as f:
        json.dump(player_payload_merged, f, ensure_ascii=False, indent=2)

    print("\n=== COPY THIS INTO YOUR MIDI PLAYER (MERGED) ===")
    print(json.dumps(player_payload_merged, ensure_ascii=False, indent=2))
    print("=== END PLAYER PAYLOAD (MERGED) ===\n")

    with open("phase2_composition.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocabulary_names_used": [v["name"] for v in vocabulary],
                "particle_names_used": [particle_by_feature[v["name"]] for v in vocabulary],
                "placements": placements,
                "runs_time_shifted": adjusted_runs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\nPhase 2 complete. Wrote time-aligned composition to phase2_composition.json")
    _maybe_export_midi(workdir)


if __name__ == "__main__":
    main()
