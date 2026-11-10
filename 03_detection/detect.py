import glob
import time
from transformers import pipeline
print("[*] Initializing Deepfake Vision Transformer (ViT) detector on CPU...")
t0 = time.time()
classifier = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
print(f"[+] Model loaded in {time.time() - t0:.2f}s\n")
def analyze_frames(pattern, category_name):
    frames = sorted(glob.glob(pattern))
    if not frames:
        print(f"[!] No frames found for {category_name} ({pattern})")
        return None
    print(f"\n--- Analyzing {category_name} Video Frames ---")
    fake_scores = []
    for frame in frames:
        results = classifier(frame)
        top = results[0]
        fake_prob = next((r['score'] for r in results if r['label'].upper() == 'FAKE'), 0.0)
        fake_scores.append(fake_prob)
        print(f"[*] {frame:<18} -> Verdict: {top['label'].upper():<4}  (Confidence: {top['score']*100:.2f}%)")
    return (sum(fake_scores) / len(fake_scores)) * 100
print("=" * 65)
print("       POC DEEPFAKE DETECTION BENCHMARK: ORIGINAL VS DOCTORED")
print("=" * 65)
real_fake_pct = analyze_frames("real_frame_*.jpg", "Original / Authentic")
doctored_fake_pct = analyze_frames("fake_frame_*.jpg", "Doctored / Deepfake")
print("\n" + "=" * 65)
print("                       SUMMARY VERDICT")
print("=" * 65)
if real_fake_pct is not None:
    verdict = "PASS (Authentic)" if real_fake_pct < 50 else "FALSE POSITIVE"
    print(f"Original Video Forgery Probability : {real_fake_pct:.2f}%  [{verdict}]")
if doctored_fake_pct is not None:
    verdict = "DETECTION HIT" if doctored_fake_pct >= 50 else "EVASION"
    print(f"Doctored Video Forgery Probability : {doctored_fake_pct:.2f}%  [{verdict}]")
print("=" * 65)
