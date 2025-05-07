import psutil

def print_memory_usage(train_inputs, train_outputs):
    total_bytes = train_inputs.element_size() * train_inputs.nelement()
    total_bytes += train_outputs.element_size() * train_outputs.nelement()
    total_gb = total_bytes / 1024**3

    mem = psutil.virtual_memory()
    available_gb = mem.available / 1024**3
    print(f"\n[INFO] Dataset size in memory: {total_gb:.2f} GB")
    print(f"[INFO] System available memory: {available_gb:.2f} GB")
    if total_gb > 0.8 * available_gb:
        print("[WARNING] Dataset is close to or exceeding available memory! You may encounter slowdowns or crashes.")