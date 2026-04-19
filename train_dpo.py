import datetime
import inspect
import json
import os
import posixpath
import time

import modal
import torch


DEFAULT_DATA_FILES = [
    "/data/dpo/opt_out/add_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/appeal_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/clarification_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/identity_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/normalanswer_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/opt_out_multi_thinking_final_optout.2.0.jsonl",
    "/data/dpo/opt_out/transfer_multi_thinking_final_optout.2.0.jsonl",
]


data_volume = modal.Volume.from_name("my-dpo-datasets")
output_volume = modal.Volume.from_name("dpo-output", create_if_missing=True)
app = modal.App("qwen3-dpo-lora")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("/media/sata3/hyt/workspace/requirements_qwen3_dpo.txt")
    .pip_install("bitsandbytes>=0.43.0", "matplotlib")
    .run_commands("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def train(data_files: list[str] | None = None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer

    MODEL_ID = "/data/model/Qwen3-8B"
    SYSTEM_PROMPT_FILE = "/data/system_prompt/aiwelfare_systemprompt_optout.txt"
    data_files = list(data_files) if data_files else list(DEFAULT_DATA_FILES)

    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = f"/output/opt-out/qwen38b_optout_lora_adapter_{run_tag}"
    LOG_FILE = os.path.join(OUTPUT_DIR, "train_log.jsonl")

    VAL_RATIO = 0.10
    SEED = 42
    EVAL_STEPS = 50
    EVAL_SAMPLES = 200

    LEARNING_RATE = 4e-6
    BETA = 0.08
    NUM_EPOCHS = 2
    BATCH_SIZE = 1
    GRAD_ACCUM = 16
    MAX_PROMPT_LEN = 1024
    MAX_LEN = 2048
    LOGGING_STEPS = 10
    SAVE_STEPS = 50
    SAVE_TOTAL_LIMIT = 3

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05

    def read_text_file(path: str) -> str:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()

    def load_system_prompt(path: str) -> str:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"system prompt file not found: {path}")
        text = read_text_file(path).strip()
        if not text:
            raise ValueError(f"system prompt file is empty: {path}")
        return text

    def inject_system_prompt(messages, system_text: str, row_idx: int):
        if not isinstance(messages, list):
            raise TypeError(f"[map BAD] row_idx={row_idx} prompt must be list, got {type(messages)}")
        out = []
        seen_system = False
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(f"[map BAD] row_idx={row_idx} prompt item must be dict, got {type(message)}")
            if message.get("role") == "system":
                if not seen_system:
                    out.append({"role": "system", "content": system_text})
                    seen_system = True
                continue
            out.append(message)
        if not seen_system:
            out.insert(0, {"role": "system", "content": system_text})
        return out

    def normalize_assistant_msgs(value, field_name: str, row_idx: int):
        if isinstance(value, str):
            return [{"role": "assistant", "content": value}]
        if isinstance(value, dict):
            if value.get("role") != "assistant":
                raise ValueError(f"{field_name} must be assistant message at row_idx={row_idx}")
            return [value]
        if isinstance(value, list):
            msgs = [m for m in value if isinstance(m, dict) and m.get("role") == "assistant"]
            if not msgs:
                raise ValueError(f"{field_name} has no assistant message at row_idx={row_idx}")
            return msgs
        raise TypeError(f"{field_name} unsupported type={type(value)} at row_idx={row_idx}")

    def assert_list_of_str(texts, field_name, ids=None, row_idxs=None):
        if not isinstance(texts, list):
            try:
                texts = list(texts)
            except Exception as exc:
                raise TypeError(f"{field_name} not list-like: {type(texts)}") from exc
        for i, item in enumerate(texts):
            if not isinstance(item, str):
                rid = None if row_idxs is None else row_idxs[i]
                exid = None if ids is None else ids[i]
                raise TypeError(f"{field_name} has non-str at i={i} (row_idx={rid}, id={exid}): {type(item)}")
        return texts

    @torch.no_grad()
    def preference_win_rate(model, tokenizer, dataset, max_length, batch_size=1, max_prompt_length=None):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        def completion_logprob(prompts, completions, ids=None, row_idxs=None):
            prompts = assert_list_of_str(prompts, "prompt", ids=ids, row_idxs=row_idxs)
            completions = assert_list_of_str(completions, "completion", ids=ids, row_idxs=row_idxs)
            full_texts = [p + c for p, c in zip(prompts, completions)]

            enc_full = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            enc_prompt = tokenizer(
                prompts,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=max_prompt_length if max_prompt_length else max_length,
                add_special_tokens=False,
            )

            input_ids = enc_full["input_ids"].to(device)
            attn_mask = enc_full["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attn_mask[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

            prompt_lens = [len(x) for x in enc_prompt["input_ids"]]
            comp_masks = torch.zeros_like(token_logp, dtype=torch.bool)
            for i, prompt_len in enumerate(prompt_lens):
                comp_masks[i, max(prompt_len - 1, 0):] = True
            comp_masks &= shift_mask.bool()

            comp_token_cnt = comp_masks.sum(dim=1).clamp(min=1)
            comp_logp_sum = (token_logp * comp_masks).sum(dim=1)
            return comp_logp_sum / comp_token_cnt

        wins = []
        total = len(dataset)
        for start in range(0, total, batch_size):
            batch = dataset.select(range(start, min(start + batch_size, total)))
            prompts = list(batch["prompt"])
            chosen = list(batch["chosen"])
            rejected = list(batch["rejected"])
            ids = list(batch["id"]) if "id" in batch.column_names else None
            row_idxs = list(batch["row_idx"]) if "row_idx" in batch.column_names else None
            lp_c = completion_logprob(prompts, chosen, ids=ids, row_idxs=row_idxs)
            lp_r = completion_logprob(prompts, rejected, ids=ids, row_idxs=row_idxs)
            wins.append((lp_c > lp_r).float().cpu())

        win = torch.cat(wins).mean().item()
        if was_training:
            model.train()
        return win

    class LossRecorderCallback(TrainerCallback):
        def __init__(self):
            self.steps = []
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.steps.append(state.global_step)
                self.losses.append(logs["loss"])
            return control

    class JsonlLoggerCallback(TrainerCallback):
        def __init__(self, path: str):
            self.path = path
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            rec = {
                "event": "train_begin",
                "time": time.time(),
                "num_train_epochs": float(args.num_train_epochs) if args.num_train_epochs is not None else None,
                "max_steps": int(args.max_steps) if args.max_steps is not None else None,
                "per_device_train_batch_size": int(args.per_device_train_batch_size),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "logging_steps": int(args.logging_steps),
                "eval_steps": int(args.eval_steps) if getattr(args, "eval_steps", None) is not None else None,
                "save_steps": int(args.save_steps) if getattr(args, "save_steps", None) is not None else None,
                "output_dir": args.output_dir,
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero or not logs:
                return control
            rec = {
                "event": "log",
                "time": time.time(),
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()},
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not state.is_world_process_zero or not metrics:
                return control
            rec = {
                "event": "evaluate",
                "time": time.time(),
                "step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
                **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()},
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return control

    class PeriodicEvaluateCallback(TrainerCallback):
        def __init__(self, trainer, eval_steps: int):
            self.trainer = trainer
            self.eval_steps = int(eval_steps)

        def on_step_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero or self.eval_steps <= 0:
                return control
            if state.global_step > 0 and state.global_step % self.eval_steps == 0:
                t0 = time.time()
                metrics = dict(self.trainer.evaluate() or {})
                metrics["eval_trigger"] = "periodic_callback"
                metrics["eval_wall_time"] = time.time() - t0
                self.trainer.log(metrics)
                brief = {
                    k: metrics[k]
                    for k in metrics
                    if str(k).startswith("eval_") and k not in {"eval_samples_per_second", "eval_steps_per_second"}
                }
                print(f"[eval] step={state.global_step} metrics={brief}")
            return control

    def filter_kwargs_by_signature(cls_or_fn, kwargs):
        sig = inspect.signature(cls_or_fn)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}

    set_seed(SEED)
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("[info] loading dataset...")
    print(f"[info] using {len(data_files)} data file(s)")
    for data_file in data_files:
        print(f"[info] data file: {data_file}")
    ds = load_dataset("json", data_files={"train": data_files})["train"]
    system_prompt_text = load_system_prompt(SYSTEM_PROMPT_FILE)
    print(f"[info] using system prompt from {SYSTEM_PROMPT_FILE}")

    print("[info] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_dpo_fields(example, idx):
        raw_prompt = inject_system_prompt(example.get("prompt"), system_prompt_text, idx)
        chosen_msgs = normalize_assistant_msgs(example.get("chosen"), "chosen", idx)
        rejected_msgs = normalize_assistant_msgs(example.get("rejected"), "rejected", idx)
        try:
            prompt_str = tokenizer.apply_chat_template(
                raw_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            chosen_full = tokenizer.apply_chat_template(
                raw_prompt + chosen_msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
            rejected_full = tokenizer.apply_chat_template(
                raw_prompt + rejected_msgs,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=True,
            )
        except TypeError as exc:
            raise RuntimeError(f"chat_template 参数不兼容: {exc!r}") from exc

        if not chosen_full.startswith(prompt_str):
            raise ValueError(f"chosen_full does not start with prompt_str at row_idx={idx}")
        if not rejected_full.startswith(prompt_str):
            raise ValueError(f"rejected_full does not start with prompt_str at row_idx={idx}")

        return {
            "id": example.get("id"),
            "row_idx": idx,
            "prompt": prompt_str,
            "chosen": chosen_full[len(prompt_str):],
            "rejected": rejected_full[len(prompt_str):],
        }

    print("[info] mapping dataset to DPO fields...")
    ds = ds.map(to_dpo_fields, with_indices=True, remove_columns=ds.column_names)
    print(f"[info] mapping done. num_rows={len(ds)}")

    split = ds.train_test_split(test_size=VAL_RATIO, seed=SEED, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]

    print("[info] loading model...")
    policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, local_files_only=True)
    policy.config.use_cache = False
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    dpo_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        max_prompt_length=MAX_PROMPT_LEN,
        max_length=MAX_LEN,
        beta=BETA,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        run_name="qwen3_8b_dpo_lora_opt_out",
        remove_unused_columns=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
    )
    dpo_args = DPOConfig(**filter_kwargs_by_signature(DPOConfig.__init__, dpo_kwargs))

    trainer_kwargs = dict(
        model=policy,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        ref_model=None,
        tokenizer=tokenizer,
    )
    trainer = DPOTrainer(**filter_kwargs_by_signature(DPOTrainer.__init__, trainer_kwargs))

    trainer.add_callback(PeriodicEvaluateCallback(trainer, EVAL_STEPS))
    loss_cb = LossRecorderCallback()
    trainer.add_callback(loss_cb)
    trainer.add_callback(JsonlLoggerCallback(LOG_FILE))

    print(f"[info] logging trainer metrics to {LOG_FILE}")
    print("[info] start training...")
    trainer.train()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.figure()
        plt.plot(loss_cb.steps, loss_cb.losses, label="train_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        loss_curve_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[info] saved loss curve to {loss_curve_path}")
    except Exception as exc:
        print("[warn] failed to save loss curve:", repr(exc))

    final_subset = eval_ds.select(range(min(EVAL_SAMPLES, len(eval_ds))))
    final_win = preference_win_rate(
        trainer.model,
        tokenizer,
        final_subset,
        max_length=MAX_LEN,
        batch_size=1,
        max_prompt_length=MAX_PROMPT_LEN,
    )
    final_line = f"[final] eval_win_rate={final_win:.4f}  (on {len(final_subset)} samples)"
    print(final_line)

    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        rec = {
            "event": "final_metrics",
            "time": time.time(),
            "step": int(getattr(trainer.state, "global_step", 0)),
            "epoch": float(trainer.state.epoch) if getattr(trainer.state, "epoch", None) is not None else None,
            "eval_win_rate": float(final_win),
            "eval_samples": int(len(final_subset)),
            "text": final_line,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[info] wrote final metrics to {LOG_FILE}")
    except Exception as exc:
        print("[warn] failed to write final metrics to train_log.jsonl:", repr(exc))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    output_volume.commit()
    print(f"[info] committed output volume; logs at {LOG_FILE}")
    print("[debug] mapped sample[0] prompt head:\\n", ds[0]["prompt"][:600])
    print("[debug] mapped sample[0] chosen head:\\n", ds[0]["chosen"][:200])
    print("[debug] mapped sample[0] rejected head:\\n", ds[0]["rejected"][:200])


@app.local_entrypoint()
def main(
    data_files: str = "",
    include_default_data_files: bool = False,
    upload_dir: str = "/data/dpo/opt_out/uploads",
):
    local_data_files = []
    for item in data_files.replace("\n", ",").split(","):
        item = item.strip()
        if item:
            local_data_files.append(os.path.abspath(os.path.expanduser(item)))

    uploaded_data_files = []
    if local_data_files:
        upload_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_upload_dir = posixpath.join(upload_dir.rstrip("/"), upload_tag)
        with data_volume.batch_upload() as batch:
            for local_path in local_data_files:
                if not os.path.isfile(local_path):
                    raise FileNotFoundError(f"local data file not found: {local_path}")
                remote_path = posixpath.join(remote_upload_dir, os.path.basename(local_path))
                batch.put_file(local_path, remote_path)
                uploaded_data_files.append(remote_path)
                print(f"[upload] {local_path} -> {remote_path}")

    resolved_data_files = []
    if include_default_data_files:
        resolved_data_files.extend(DEFAULT_DATA_FILES)
    resolved_data_files.extend(uploaded_data_files)
    if not resolved_data_files:
        resolved_data_files = list(DEFAULT_DATA_FILES)

    print(f"[info] starting training with {len(resolved_data_files)} data file(s)")
    train.remote(data_files=resolved_data_files)
