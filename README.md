# Project Overview

This project integrates OpenAI Gym's Taxi-v3 environment with the fully local **Qwen3 0.6B** model for decision-making and exposes an immersive web dashboard to watch the agent plan in real time. The backend is pure Python (Flask + Gymnasium) so you can keep extending or fine‑tuning the policy pipeline.

## Project Structure

```
gym-taxi-qwen3
├── backend
│   ├── __init__.py
│   ├── main.py
│   ├── llm
│   │   ├── __init__.py
│   │   └── client.py
│   ├── taxi
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── prompt_builder.py
│   │   └── state_utils.py
├── config
│   ├── __init__.py
│   └── settings.py
├── frontend
│   ├── static
│   │   ├── css
│   │   │   └── styles.css
│   │   └── js
│   │       └── app.js
│   └── templates
│       └── index.html
├── tests
│   ├── __init__.py
│   └── test_state_utils.py
├── .env.example
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gym-taxi-qwen3.git
   cd gym-taxi-qwen3
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install PyTorch first (choose the command that matches your platform from [pytorch.org](https://pytorch.org/get-started/locally/)), then install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment defaults and adjust if needed:

   ```bash
   cp .env.example .env
   ```

   Leave `QWEN_DEVICE` blank to let Transformers pick the best device automatically (CPU, CUDA, or Apple Silicon MPS), or set it explicitly (e.g. `cuda`, `mps`, or `0`).
   On macOS we default to CPU for stability; set `QWEN_DEVICE=mps` to force Apple GPU acceleration once you’ve confirmed it behaves well on your machine.

5. (Optional but recommended) Warm the Hugging Face cache so the first web request is instant:

   ```bash
   python - <<'PY'
   from transformers import pipeline
   pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", trust_remote_code=True)
   pipe([{ "role": "user", "content": "ping" }], max_new_tokens=4)
   PY
   ```
   This downloads the 0.6B checkpoint into `~/.cache/huggingface/hub`.

## Running the Application

1. Start the backend server (Flask will also serve the static frontend):

   ```bash
   python backend/main.py
   ```

2. Navigate to `http://localhost:5001` to open the dashboard. Hit **Reset** once to spin up the environment and model.

3. Use **Step once** to watch a single decision or **Start auto-play** to let Qwen repeatedly act until the episode ends. The timeline panel shows each action, the incremental reward, and the model's reasoning trace.

   > ℹ️ The first request may take a minute while Hugging Face downloads the 0.6 B checkpoint. During that warm-up the UI will show a fallback move; once the model loads, decisions become instant.

## Usage

- Fully local inference — nothing leaves your machine. You can tweak the prompt or swap models without touching third-party APIs.
- Detailed state summaries plus the raw ANSI render from Gymnasium help you verify each move visually.
- The history panel keeps the last dozen decisions so you can debug chains of thought before fine-tuning.

## Testing

To make sure the helper utilities behave correctly, run:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.