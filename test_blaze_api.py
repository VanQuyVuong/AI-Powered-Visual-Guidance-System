# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import requests, json, base64, time

API_KEY = "68f5578def0d2c60d7cd49e9743f0c27ede87304"
H = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

payload = {
    "query": "Can than co xe may phia truoc.",
    "speaker_id": "HN-Nar-Nu-PhuongLy",
    "model": "v2.0_flash",
    "language": "vi",
    "audio_format": "mp3",
}
r = requests.post("https://api.blaze.vn/v1/tts", json=payload, headers=H, timeout=15)
data = r.json()
job_id = data.get("id")
corr_id   = r.headers.get("x-blaze-tts-correlation-id","")
gw_corr   = r.headers.get("x-blaze-gateway-correlation-id","")
print(f"Job ID    : {job_id}")
print(f"Corr ID   : {corr_id}")
print(f"GW Corr   : {gw_corr}")

time.sleep(4)

# Try correlation IDs + extra patterns
patterns = [
    f"https://api.blaze.vn/v1/tts/{corr_id}",
    f"https://api.blaze.vn/v1/tts/{gw_corr}",
    f"https://api.blaze.vn/v1/audio/{corr_id}",
    f"https://api.blaze.vn/v1/audio/{corr_id}.mp3",
    # Try with query param
    f"https://api.blaze.vn/v1/tts?id={job_id}",
    f"https://api.blaze.vn/v1/tts?job_id={job_id}",
    # Try jobs endpoint
    f"https://api.blaze.vn/v1/jobs/{job_id}",
    f"https://api.blaze.vn/v1/jobs/{corr_id}",
    # Try tts/sync for synchronous endpoint
    "https://api.blaze.vn/v1/tts/sync",
    "https://api.blaze.vn/v1/tts/synchronous",
]

for ep in patterns:
    try:
        if ep.startswith("https://api.blaze.vn/v1/tts/sync"):
            resp = requests.post(ep, json=payload, headers=H, timeout=15)
        else:
            resp = requests.get(ep, headers=H, timeout=8)
        ct = resp.headers.get("Content-Type","")
        if resp.status_code == 200:
            if "audio" in ct:
                print(f"SUCCESS AUDIO! {ep} -> {len(resp.content)} bytes")
                with open("blaze_audio.mp3","wb") as f:
                    f.write(resp.content)
            elif "json" in ct:
                d = resp.json()
                print(f"SUCCESS JSON {ep}: {json.dumps(d, ensure_ascii=False)[:250]}")
            else:
                print(f"200 {ct}: {ep} size={len(resp.content)}")
        elif resp.status_code != 404:
            print(f"{resp.status_code} {ep}: {resp.text[:80]}")
        # silently skip 404s
    except Exception as e:
        print(f"ERR {ep}: {e}")

# Try SYNCHRONOUS mode: omit async fields, add sync flag
print("\n--- Try sync options ---")
sync_tests = [
    {"query":"Test.","speaker_id":"HN-Nar-Nu-PhuongLy","model":"v2.0_flash","language":"vi","audio_format":"mp3","sync":True},
    {"query":"Test.","speaker_id":"HN-Nar-Nu-PhuongLy","model":"v2.0_flash","language":"vi","audio_format":"mp3","async":False},
    {"query":"Test.","speaker_id":"HN-Nar-Nu-PhuongLy","model":"v2.0_flash","language":"vi","audio_format":"mp3","return_audio":True},
    {"query":"Test.","speaker_id":"HN-Nar-Nu-PhuongLy","model":"v1.5_flash","language":"vi","audio_format":"mp3"},  # older model sync?
]
for p in sync_tests:
    time.sleep(1.5)
    resp = requests.post("https://api.blaze.vn/v1/tts", json=p, headers=H, timeout=15)
    ct = resp.headers.get("Content-Type","")
    if resp.status_code == 200 and "audio" in ct:
        print(f"SYNC SUCCESS! {list(p.keys())} -> {len(resp.content)} bytes")
        with open("blaze_sync.mp3","wb") as f: resp.content
    elif "json" in ct:
        d = resp.json()
        print(f"{resp.status_code} {list(p.keys())}: {json.dumps(d,ensure_ascii=False)[:120]}")
    else:
        print(f"{resp.status_code} {resp.text[:80]}")
