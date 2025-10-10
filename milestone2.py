
import cv2
import base64
import numpy as np
import pandas as pd
from deepface import DeepFace
from google.colab import output
from typing import List, Dict, Tuple
import os, glob, json, pickle, math, uuid, shutil
from IPython.display import Javascript, display, HTML

MODEL_NAME = "ArcFace"            # robust & fast for verification
DETECTOR   = "retinaface"         # accurate detector (ships with DeepFace)


def tree(path, level=2):
    for root, dirs, files in os.walk(path):
        depth = root.replace(path, "").count(os.sep)
        if depth > level: 
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        for f in files[:20]:
            print(f"{indent}  {f}")

def read_bgr(img_or_path):
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_or_path}")
        return img
    elif isinstance(img_or_path, np.ndarray):
        return img_or_path
    else:
        raise TypeError("Provide file path or numpy array")

def get_embedding(img_or_path):
    """
    Returns L2-normalized embedding (np.array) for the largest detected face.
    """
    reps = DeepFace.represent(
        img_path = img_or_path,
        model_name = MODEL_NAME,
        detector_backend = DETECTOR,
        enforce_detection = True,
        align = True,
        normalization = "ArcFace"  # ensures embeddings are already normalized
    )
    # DeepFace.represent returns a list (one per detected face); use the first
    if not isinstance(reps, list) or len(reps) == 0:
        raise ValueError("No face embedding returned")
    emb = np.array(reps[0]["embedding"], dtype=np.float32)
    # ArcFace reps are already normalized, but normalize again just in case
    n = np.linalg.norm(emb) + 1e-12
    return emb / n

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))


def list_images(root):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True) if p.lower().endswith(exts)]

def build_enrollment_db(enroll_dir, out_path):
    rows = []
    persons = sorted([d for d in os.listdir(enroll_dir) if os.path.isdir(os.path.join(enroll_dir, d))])
    if not persons:
        raise RuntimeError(f"No person folders found under {enroll_dir}. Create enroll/<person>/images")
    for person in persons:
        pdir = os.path.join(enroll_dir, person)
        imgs = list_images(pdir)
        if not imgs:
            print(f"WARNING: No images for {person} — skipping")
            continue
        for img_path in imgs:
            try:
                emb = get_embedding(img_path)
                rows.append({"person": person, "img_path": img_path, "embedding": emb})
                print(f"[OK] {person} <- {os.path.basename(img_path)}")
            except Exception as e:
                print(f"[ERR] {img_path}: {e}")

    if not rows:
        raise RuntimeError("No embeddings created — check your images.")
    df = pd.DataFrame(rows)
    # Save as pickle (embeddings as float32 arrays)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    print(f"\nSaved DB with {len(df)} entries for {df['person'].nunique()} persons -> {out_path}")
    return df

def build_person_prototypes(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    protos = {}
    for person, g in df.groupby("person"):
        embs = np.stack(g["embedding"].to_list(), axis=0)
        proto = embs.mean(axis=0)
        proto /= (np.linalg.norm(proto)+1e-12)
        protos[person] = proto.astype(np.float32)
    return protos

def identify_image(img_or_path, person_protos, threshold) -> Tuple[str, float]:
    """
    Returns (best_person_or_'Unknown', best_similarity)
    """
    q = get_embedding(img_or_path)
    best_p, best_s = "Unknown", -1.0
    for p, proto in person_protos.items():
        s = cosine_sim(q, proto)
        if s > best_s:
            best_s, best_p = s, p
    if best_s >= threshold:
        return best_p, best_s
    return "Unknown", best_s

def record_webcam(out_path, seconds=5, show_preview=False):
    js = Javascript(r"""
    async function recordVideo(seconds) {
      const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: true});
      const rec = new MediaRecorder(stream, {mimeType: 'video/webm'});
      let data = [];
      rec.ondataavailable = e => data.push(e.data);
      rec.start();

      // Simple on-screen timer
      const div = document.createElement('div');
      div.style = 'position:fixed;top:16px;left:16px;background:#0008;color:#fff;padding:8px 12px;border-radius:8px;z-index:9999;font:14px sans-serif';
      document.body.appendChild(div);

      let t = seconds;
      const tick = setInterval(()=>{ div.textContent = `Recording... ${t}s`; t--;}, 1000);

      await new Promise(r => setTimeout(r, seconds * 1000));
      rec.stop();
      await new Promise(r => rec.onstop = r);
      clearInterval(tick);
      document.body.removeChild(div);

      stream.getTracks().forEach(t => t.stop());
      const blob = new Blob(data, {type: 'video/webm'});
      const reader = new FileReader();
      reader.readAsDataURL(blob);
      await new Promise(r => reader.onloadend = r);
      return reader.result; // dataURL
    }
    """)
    display(js)
    data_url = output.eval_js(f"recordVideo({int(seconds)})")

    header, encoded = data_url.split(',', 1)
    binary = base64.b64decode(encoded)
    with open(out_path, "wb") as f:
      f.write(binary)

    if show_preview:
      display(HTML(f"""
      <video src="recorded.webm" controls playsinline style="max-width: 480px; border-radius:12px;"></video>
      <p style="font:14px/1.4 sans-serif">Saved: <code>{out_path}</code></p>
      """))
    return out_path