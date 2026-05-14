import torch

def gpu_capability_report():
    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
        return

    props = torch.cuda.get_device_properties(0)
    major, minor = props.major, props.minor
    sm = f"{major}.{minor}"

    print(f"GPU:              {props.name}")
    print(f"VRAM:             {props.total_memory / 1e9:.2f} GB")
    print(f"SM version:       {sm}  (Compute Capability)")
    print(f"CUDA cores:       {props.multi_processor_count} SMs")
    print()

    # --- Precision support ---
    bf16  = major >= 8                          # Ampere and above
    fp16  = major >= 6                          # Pascal and above
    int8  = major >= 6
    tf32  = major >= 8
    fp8   = major > 8 or (major == 8 and minor >= 9)  # Ada Lovelace / Hopper

    print("=== Precision Support ===")
    print(f"  FP32:   always supported")
    print(f"  FP16:   {'YES' if fp16 else 'NO'}  (safe for inference; unstable for LLM training)")
    print(f"  BF16:   {'YES — use this for training' if bf16 else 'NO — use FP16 with grad scaling'}")
    print(f"  TF32:   {'YES — enabled by default in PyTorch' if tf32 else 'NO'}")
    print(f"  INT8:   {'YES' if int8 else 'NO'}  (post-training quantization / bitsandbytes)")
    print(f"  FP8:    {'YES' if fp8 else 'NO'}  (H100/RTX 4090 only)")
    print()

    # --- Training techniques ---
    flash_attn = major >= 8
    grad_ckpt  = True   # always supported in software
    fsdp       = torch.cuda.device_count() > 1
    nf4_qlora  = major >= 7   # bitsandbytes requires Turing+

    print("=== Training Technique Support ===")
    print(f"  Full finetuning:         always possible (VRAM permitting)")
    print(f"  LoRA:                    always possible")
    print(f"  QLoRA (NF4):             {'YES' if nf4_qlora else 'NO — need Turing (SM 7.x)+'}")
    print(f"  Gradient checkpointing:  YES (software, always)")
    print(f"  Flash Attention 2:       {'YES' if flash_attn else 'NO — need Ampere (SM 8.x)+'}")
    print(f"  Multi-GPU (FSDP):        {'YES — ' + str(torch.cuda.device_count()) + ' GPUs' if fsdp else 'NO — single GPU only'}")
    print()

    # --- VRAM budget ---
    vram = props.total_memory / 1e9
    print("=== What Fits in Your VRAM ===")
    models = [
        ("1B  model inference (FP16)",   2),
        ("1B  model QLoRA training",      4),
        ("3B  model inference (4-bit)",   3),
        ("3B  model QLoRA training",      8),
        ("7B  model inference (4-bit)",   5),
        ("7B  model QLoRA training",     10),
        ("13B model inference (4-bit)",   9),
        ("13B model QLoRA training",     16),
        ("70B model inference (4-bit)",  40),
    ]
    for label, required in models:
        status = "YES" if vram >= required else "NO "
        print(f"  {status}  {label:40s} (needs ~{required} GB)")

gpu_capability_report()
