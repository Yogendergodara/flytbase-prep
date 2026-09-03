"""#28: retry wrapper for the network-exposed calls in this repo - model
weight downloads (Ultralytics/HF Hub) and video source opens. Not a general
resilience framework: three attempts, linear backoff, then raise. Anything
that fails for a reason retrying won't fix (bad weights path, corrupt file)
still fails immediately after the retries - this only helps transient
network hiccups, which is what "no retries around model downloads" actually
meant in practice on a shared venue wifi.
"""
import time


def retry(fn, *args, attempts=3, delay=2.0, label="call", **kwargs):
    last_err = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            print(f"[retry] {label} failed (attempt {i + 1}/{attempts}): {e}")
            if i < attempts - 1:
                time.sleep(delay * (i + 1))
    raise last_err
