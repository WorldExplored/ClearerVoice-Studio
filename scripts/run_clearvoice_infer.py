import argparse
import json
import os
import time
from datetime import datetime

import soundfile as sf

from clearvoice import ClearVoice


def seconds_in_file(wav_path: str) -> float:
    info = sf.info(wav_path)
    if info.samplerate == 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def run_infer(model_key: str, input_path: str, output_dir: str = None):
    if model_key == "se48":
        task = "speech_enhancement"
        model_names = ["MossFormer2_SE_48K"]
        display_name = "MossFormer2_SE_48K"
    elif model_key == "ss16":
        task = "speech_separation"
        model_names = ["MossFormer2_SS_16K"]
        display_name = "MossFormer2_SS_16K"
    elif model_key == "sr48":
        task = "speech_super_resolution"
        model_names = ["MossFormer2_SR_48K"]
        display_name = "MossFormer2_SR_48K"
    else:
        raise ValueError("model_key must be one of: se48, ss16, sr48")

    # Init model (download checkpoints on first run)
    model_init_t0 = time.time()
    cv = ClearVoice(task=task, model_names=model_names)
    model_init_t1 = time.time()

    # Inference (single file path expected)
    infer_t0 = time.time()
    outputs = cv(input_path=input_path, online_write=False)
    infer_t1 = time.time()

    # Metrics
    # tpot: total inference time for this single call
    tpot = infer_t1 - infer_t0
    # ttft: time-to-first-output; for single-file synchronous API it's equal to tpot
    ttft = tpot

    # processed seconds: prefer output length if accessible; otherwise use input length
    try:
        model_sr = cv.models[0].args.sampling_rate
        if isinstance(outputs, list):
            # Separation returns list of [spk arrays]; use time length from first speaker
            if len(outputs) > 0 and hasattr(outputs[0], 'shape') and len(outputs[0].shape) > 0:
                out_len = int(outputs[0].shape[-1])
            else:
                out_len = 0
        else:
            # Enhancement / SR returns ndarray; use last dimension
            out_len = int(outputs.shape[-1])
        processed_seconds = (out_len / float(model_sr)) if (model_sr and out_len) else seconds_in_file(input_path)
    except Exception:
        processed_seconds = seconds_in_file(input_path)

    tps = processed_seconds / tpot if tpot > 0 else 0.0

    # Optional write (disabled by default)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Delegate file writing to ClearVoice to preserve input container/format
        out_path = os.path.join(output_dir, f"output_{display_name}.wav")
        cv.write(outputs, output_path=out_path)

    # Log record
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": display_name,
        "task": task,
        "input_path": os.path.abspath(input_path),
        "ttft": ttft,
        "tpot": tpot,
        "tps": tps,
        "model_init_s": model_init_t1 - model_init_t0,
        "processed_seconds": processed_seconds,
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Run ClearVoice pretrained models and log ttft/tpot/tps")
    parser.add_argument("--model", required=True, choices=["se48", "ss16", "sr48"], help="Which model to run: se48|ss16|sr48")
    parser.add_argument("--input", required=True, help="Path to a single input audio file")
    parser.add_argument("--write_dir", default=None, help="Optional directory to save output audio")
    parser.add_argument("--log_dir", default="logs", help="Directory to write JSONL logs")
    args = parser.parse_args()

    rec = run_infer(args.model, args.input, args.write_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"{args.model}_inference.jsonl")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    # Also print concise summary
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()


