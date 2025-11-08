from pathlib import Path

from flask import Flask, render_template, request

# Support both "python -m src.app" and "python src/app.py"
try:
    from .inference import load_artifacts, predict_text  # type: ignore
except ImportError:  # pragma: no cover
    from inference import load_artifacts, predict_text  # type: ignore


app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent.parent / "templates"),
    static_folder=str(Path(__file__).parent.parent / "static"),
)

# Load model and vocab at startup
MODELS_DIR = Path(__file__).parent.parent / "models"
VOCAB_PATH = Path(__file__).parent.parent / "data" / "processed" / "vocab.json"
MODEL_PATH = MODELS_DIR / "model.joblib"

VOCAB, MODEL = None, None
if VOCAB_PATH.exists() and MODEL_PATH.exists():
    try:
        VOCAB, MODEL = load_artifacts(VOCAB_PATH, MODEL_PATH)
    except Exception:
        VOCAB, MODEL = None, None


@app.get("/health")
def health() -> tuple[str, int]:
    return "ok", 200


@app.route("/", methods=["GET", "POST"])
def index():
    error: str | None = None
    result: dict | None = None
    text: str = ""

    if request.method == "POST":
        text = (request.form.get("email") or "").strip()
        if not text:
            error = "Please paste an email to analyze."
        else:
            if VOCAB is None or MODEL is None:
                error = "Model artifacts not found. Train the model first."
            else:
                label, prob, explanation = predict_text(text, VOCAB, MODEL)
                result = {
                    "label": "phishing" if label == 1 else "legitimate",
                    "confidence": prob,
                    "note": None,
                    "explanation": explanation,
                }

    return render_template("index.html", text=text, result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


