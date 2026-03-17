# openai_helper.py
from openai import OpenAI
import os
import base64
from typing import Optional

# Default client uses environment-configured OpenAI creds
client = OpenAI()

def chat_explain(prompt: str) -> str:
    """
    Send prompt to GPT-4o-mini and return the explanation text.
    Returns empty string on error.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant speaking gently to concerned parents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=350
        )
        content = response.choices[0].message.content
        return (content or "").strip()
    except Exception as e:
        print("OpenAI error in chat_explain:", str(e))
        return ""


def explain_with_gradcam(patient_img_path: str, gradcam_img_path: str,
                         model: str = "qwen2.5vl:3b",
                         base_url: Optional[str] = None,
                         api_key: Optional[str] = None) -> str:
    """
    Send patient image + grad-cam heatmap to a local OpenAI-style server and
    return the assistant's short explanation string. Uses a local client when
    `base_url` is provided (defaults to http://localhost:11434/v1).
    """
    def img_to_b64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    patient_b64 = img_to_b64(patient_img_path)
    gradcam_b64 = img_to_b64(gradcam_img_path)

    # If a custom base_url is provided, create a temporary client pointing there.
    use_client = client
    created_temp = False
    if base_url:
        created_temp = True
        use_client = OpenAI(base_url=base_url, api_key=(api_key or "ollama"))
    else:
        # fallback to localhost if no base_url provided and env not set
        env_base = os.environ.get("LOCAL_OPENAI_BASE_URL")
        if env_base:
            created_temp = True
            use_client = OpenAI(base_url=env_base, api_key=(api_key or os.environ.get("LOCAL_OPENAI_API_KEY", "ollama")))

    # Try the requested model first, then fall back to smaller models when
    # Ollama reports memory/VRAM limits. The fallback list can be overridden
    # via the OLLAMA_FALLBACK_MODELS env var as a comma-separated list.
    fallback_env = os.environ.get("OLLAMA_FALLBACK_MODELS", "qwen2.5vl:3b,qwen2-mini")
    candidates = [model] + [m for m in [s.strip() for s in fallback_env.split(",")] if m and m != model]

    def _call_model(m: str) -> Optional[str]:
        try:
            response = use_client.chat.completions.create(
                model=m,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are a medical AI for explainable Down syndrome screening from facial images.\n"
                                    "Analyze the original face photo and the Grad-CAM heatmap (red areas = strongest model attention).\n"
                                    "Provide a short, professional, factual explanation (4-7 sentences):\n"
                                    "- Key facial regions with high attention (e.g., nasal bridge, eyes, ears)\n"
                                    "- How these relate to common Down syndrome facial characteristics\n"
                                    "- This is AI-based screening only — not a diagnosis; recommend genetic testing if concerned.\n"
                                    "Keep tone neutral, accurate, and ethical. No overconfidence."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{patient_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gradcam_b64}"}}
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3,
            )

            # Support both new and legacy response shapes
            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                content = getattr(response.choices[0], "text", None)
            return (content or "").strip()
        except Exception as e:
            msg = str(e)
            print(f"Error calling model '{m}': {msg}")
            # Heuristic: if error mentions memory/VRAM, try the next candidate
            low_memory_indicators = ["memory", "vram", "system memory", "requires more", "out of memory"]
            if any(tok in msg.lower() for tok in low_memory_indicators):
                print(f"Model '{m}' appears to exceed available memory; trying next fallback model")
                return None
            # For other errors, log and return None so we can try other fallbacks
            return None

    for candidate in candidates:
        result = _call_model(candidate)
        if result:
            # cleanup if we created a temporary client
            if created_temp:
                try:
                    del use_client
                except Exception:
                    pass
            return result

    # If all image+model attempts failed, try a text-only fallback to the
    # smallest candidate (last in the list). This avoids sending large image
    # blobs when system memory is constrained.
    try_text_only = os.environ.get("OLLAMA_ALLOW_TEXT_ONLY", "1") == "1"
    if try_text_only and candidates:
        small_model = candidates[-1]
        try:
            print(f"Attempting text-only fallback with model '{small_model}'")
            response = use_client.chat.completions.create(
                model=small_model,
                messages=[
                    {"role": "system", "content": "You are a concise, factual medical assistant."},
                    {"role": "user", "content": (
                        "Images could not be processed due to system memory limits. "
                        "Based on typical Grad-CAM attention patterns over face photos (e.g., attention on nasal bridge, eyes, ears), "
                        "provide a short (3-5 sentence) neutral explanation of what high attention in those regions may suggest for screening purposes, "
                        "and recommend next steps such as clinical evaluation and genetic testing if there is concern. "
                        "Do not assert a diagnosis."
                    )}
                ],
                max_tokens=220,
                temperature=0.25,
            )
            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                content = getattr(response.choices[0], "text", None)

            if content:
                if created_temp:
                    try:
                        del use_client
                    except Exception:
                        pass
                return content.strip()
        except Exception as e:
            print("Text-only fallback failed:", str(e))

    print("All explain_with_gradcam model attempts and text-only fallback failed or returned no content")
    if created_temp:
        try:
            del use_client
        except Exception:
            pass
    return ""