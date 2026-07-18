import json
import argparse
import numpy as np

def apply_gating(in_path, out_path):
    with open(in_path, 'r') as f_in, open(out_path, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            if "pred_exist_logits" in data:
                logits = data["pred_exist_logits"]
                pred_c = int(np.argmax(logits))
                
                if pred_c == 0:
                    data["pred_relevant_windows"] = []
                    data["pred_exist_score"] = 0.0
                else:
                    windows = data["pred_relevant_windows"][:pred_c]
                    if pred_c >= 3 and len(windows) > 0:
                        top1_score = windows[0][2]
                        thd = max(0.5, 0.8 * top1_score)
                        filtered_windows = []
                        for i, w in enumerate(windows):
                            if i < 2 or w[2] >= thd:
                                filtered_windows.append(w)
                        windows = filtered_windows
                    data["pred_relevant_windows"] = windows
                    data["pred_exist_score"] = 1.0
            f_out.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply AMC gating to predictions")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    
    apply_gating(args.input_path, args.output_path)
    print(f"Gating applied. Output saved to {args.output_path}")
