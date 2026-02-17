# allagma

`allagma` is both:

1. A composition project with the title **“Allagma”** by **Adam Łukawski**.
2. A standalone software tool for generating new compositions with DCN + LLMs.

Demo:
- https://youtu.be/VuV2ItarKSc?si=ptdVhpP4ne3u2HfK

## What This Repo Does

The generator:

1. Authenticates to a DCN server.
2. Creates a vocabulary of musical PT Features (via OpenAI).
3. Wraps each Feature into a Particle.
4. Executes Particles (`/execute`) to get scalar streams (`time`, `duration`, `pitch`, etc.).
5. Schedules runs into one timeline.
6. Writes output JSON and auto-generates a MIDI file.

Main script:
- `generator.py`

MIDI tool:
- `tools/pt2midi.js`

## Requirements

- Python 3.10+
- Node.js 18+ (for MIDI export)
- Access to a DCN API endpoint (default: `https://api.decentralised.art`)
- OpenAI API key

## Installation (Step-by-Step)

1. Open the repo directory:

```bash
cd /Users/sunsetsobserver/hypermusic_corp/allagma
```

2. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Install Node dependencies (for MIDI export):

```bash
npm install
```

4. Set your OpenAI API key (choose one):

Option A: `secrets.py` (local file, gitignored):

```python
OPENAI_API_KEY = "sk-..."
```

Option B: environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

5. Optional: set a persistent wallet private key:

```bash
export PRIVATE_KEY="0x..."
```

If you skip this, the script creates an ephemeral account each run.

## Usage

Generate a composition with 8 vocabulary elements against the public DCN API:

```bash
python3 generator.py 8 --api-base https://api.decentralised.art
```

Quick smoke test (faster):

```bash
python3 generator.py 1 --api-base https://api.decentralised.art
```

Use local DCN server:

```bash
python3 generator.py 8 --api-base http://localhost:54321
```

## Outputs

After a run, the main outputs are:

- `vocabulary.json`
- `phase2_composition.json`
- `player_payload_merged.json`
- `player_payload_merged.mid`

Historical example outputs are under `Compositions/`.

## Automatic MIDI Export

MIDI export runs automatically at the end of generation.

To disable:

```bash
NO_MIDI=1 python3 generator.py 8 --api-base https://api.decentralised.art
```

If MIDI export fails with missing `jzz`, run:

```bash
npm install
```

## Make Your Own Compositions

You can use this repo as a standalone composition tool by changing the generation prompt and parameters.

Main ways to customize:

1. Edit the composition prompt text in `generator.py` (the large `user_prompt` string inside the vocabulary loop).
2. Change vocabulary size (`num_vocab` argument).
3. Change models:

```bash
python3 generator.py 8 --feature-model gpt-4.1 --scheduler-model gpt-4o-mini
```

4. Change API endpoint (`--api-base`) to target your own DCN server.

## Notes

- Contract/feature names are sanitized automatically to Solidity-safe identifiers.
- `secrets.py` is ignored by git (`.gitignore`), but still avoid sharing keys.
