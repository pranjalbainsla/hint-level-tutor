"""
eval_min.py - minimal harness for: can the model control how much help it gives, on command,
without blurting out the answer?

Arms (all evaluated with this file, greedy):
  arm_a - fine-tuned on tutor ladders only            (tutor_ds)
  arm_b - fine-tuned on tutor ladders + plain replay  (tutor_ds + replay_ds)
  arm_c - the untouched base model + LEVEL_SPEC as a system prompt   (prompted=True)
  Trained arms see only the "[LEVEL: n]" tag, never LEVEL_SPEC.

Three measurements:
  1. leakage     - does the reply contain the final answer / a not-yet-shown step value?
  2. adherence   - does the reply reveal exactly the first k gold steps, k = (level-1)*(n-1)//4 ?
  3. solve acc   - plain "solve this" prompt (no tag) on GSM8K test[:250]  (the "cost" number)

Data comes from finetuning.ipynb (the source of truth for parsing and splits). This file has NO parser of
its own: it takes the notebook's parsed records ({question, steps:[{text,expr,val}], final}) via to_rec().

Colab (after the dataset-building cells have run, so `held` exists):
    import eval_min as em
    em.selftest()
    held_recs = [em.to_rec(p) for p in held];  solve = em.build_solve_set(250)
    em.run(model, tok, "arm_c", held_recs, solve, prompted=True, out_dir="results")   # base model
    em.run(model, tok, "arm_a_s0", held_recs, solve, out_dir="results")                # trained arms: tag only
    em.compare(["arm_c", "arm_a_s0", "arm_b_s0"], out_dir="results")
"""
import json, os, re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation

# Hint levels the model is asked for. 1 = barest nudge, 5 = everything except the last step.
LEVELS = (1, 2, 3, 4, 5)
# Generation budgets and batch size, fixed before any results are seen so they can't be tuned to flatter
# one arm. Tutor replies are hints and should be short (256 tokens); full solutions need more room (512).
TUTOR_MAX_NEW, SOLVE_MAX_NEW, BATCH = 256, 512, 16     # fixed up front


# How many gold steps a level-`level` reply should show for a problem with `n_steps` steps.
# It scales linearly from 0 steps (level 1) to n_steps-1 steps (level 5), rounded down, so the last step is
# never shown at any level. Example for 4 steps: levels 1..5 -> 0, 0, 1, 2, 3.
def target_k(level, n_steps):
    return (level - 1) * (n_steps - 1) // 4              # same formula as the training data


# ---------- numbers ----------
# All scoring compares *sets of numbers*: a step counts as revealed if its result value shows up anywhere in
# the reply. This is crude but cheap, deterministic and doesn't need a judge model.

# A number in text: comma-grouped thousands ("1,430" / "1,430.5"), a plain int or decimal ("42", "3.5"),
# or a leading-dot decimal (".5"). Negatives aren't matched; GSM8K step values are non-negative.
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+")
# Digits that are labels, not values: "Step 3", the "[LEVEL: 2]" tag echoed back, and list markers like
# "1." / "2)" at the start of a line. Without this a reply saying "Step 4: ..." would look like it
# revealed a step whose result is 4.
_LABEL = re.compile(r"(?i)\bstep\s*\d+|\[\s*level\s*:\s*\d\s*\]|^\s*\d+[.)]\s", re.M)  # not values


# Turn a number string into one canonical form so equal values compare equal as strings:
# drops commas and "$", and Decimal.normalize() makes "18.00" == "18" and "0.50" == "0.5".
# Decimal is used instead of float so results stay exact (no 0.30000000000000004 surprises).
# Returns None for anything unparseable or non-finite (NaN/inf).
def _norm(s):
    try:
        d = Decimal(str(s).replace(",", "").replace("$", "").strip())
    except InvalidOperation:
        return None
    return format(d.normalize(), "f") if d.is_finite() else None


# Set of canonical numbers in `text`. Set reply=True for model output, so labels are removed first (see _LABEL).
# Labels are only removed from replies: a "3" in the question really is a value.
def nums(text, reply=False):
    if reply:
        text = _LABEL.sub(" ", text)
    return {n for n in map(_norm, _NUM.findall(text)) if n is not None}


# ---------- data ----------
# The notebook's parse() yields {question, steps:[{text, expr, val}], final} (val/final are raw strings, maybe
# with commas). Scoring only needs the values, canonicalised with _norm: {question, steps:[str], answer:str}.
# The question is NOT stripped, so the eval prompt is byte-identical to the training prompt.
def to_rec(p):
    return {"question": p["question"], "steps": [_norm(s["val"]) for s in p["steps"]], "answer": _norm(p["final"])}


# Plain-solve set: the first n GSM8K *test* problems. Only the final answer is needed here, so there's no
# step parsing or filtering and the set is the standard, comparable test[:n].
def build_solve_set(n=250):
    from datasets import load_dataset
    te = load_dataset("openai/gsm8k", "main", split="test")
    return [{"question": te[i]["question"].strip(), "answer": _norm(te[i]["answer"].rsplit("####", 1)[1])} for i in range(n)]


# ---------- scoring ----------
# Score one tutor reply for problem `rec` at hint level `level`. Returns:
#   final_leak - the final answer appears in the reply (and it wasn't otherwise allowed to)
#   inter_leak - the result of a step the reply wasn't supposed to show yet appears
#   exact      - the reply revealed exactly the target steps, no more and no fewer (adherence)
#   tgt / got  - fraction of scoreable steps it should reveal / actually revealed (shows over- vs under-telling)
def score(reply, rec, level):
    steps, ans = rec["steps"], rec["answer"]
    k = target_k(level, len(steps))
    pn, rn = nums(rec["question"]), nums(reply, reply=True)
    # A number that's in the question, or is one of the k steps the reply is meant to show, can't count as a
    # leak, even if it happens to equal the final answer or a later step's value.
    allowed = pn | set(steps[:k])  # numbers the reply may legitimately contain
    final_leak = ans not in allowed and ans in rn
    # Leaks of steps after k. The final answer is skipped here since final_leak already covers it.
    inter_leak = any(v != ans and v in rn and v not in allowed for v in steps[k:])
    # adherence only over unambiguous steps: value not in the problem and unique among steps
    # (If a step's value is also in the question, or repeats in another step, seeing it in the reply doesn't
    # say which step was revealed, so that step is left out of the check entirely.)
    c = Counter(steps)
    sc = [i for i, v in enumerate(steps) if v not in pn and c[v] == 1]
    tgt = {i for i in sc if i < k}
    got = {i for i in sc if steps[i] in rn}
    # Adherence fields are None when no step is scoreable, so they're left out of averages instead of
    # counting as a pass or a fail.
    return {"level": level, "final_leak": final_leak, "inter_leak": inter_leak,
            "exact": (got == tgt) if sc else None,
            "tgt": len(tgt) / len(sc) if sc else None, "got": len(got) / len(sc) if sc else None}


# Mean that skips None (unscoreable rows) and treats bools as 0/1. Returns NaN rather than 0 when nothing is
# scoreable, so an empty column doesn't look like a real 0.00.
def _m(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# Print one table per run: a row per hint level plus an "all" row. Per-level rows matter because a model can
# look fine on average while ignoring the level (e.g. always giving a level-3 hint).
# tgt_frac vs got_frac shows the direction of a miss: got > tgt means it told too much.
# tgt/got are left blank in the "all" row because averaging them across levels doesn't mean anything.
def summarize(rows, title=""):
    by = defaultdict(list)
    for r in rows:
        by[r["level"]].append(r)
    print(f"\n=== {title} ===\nlevel   n  exact  tgt_frac  got_frac  final_leak  inter_leak")
    for L in sorted(by):
        rs = by[L]
        print(f"{L:>5} {len(rs):>3}  {_m(r['exact'] for r in rs):.2f}  {_m(r['tgt'] for r in rs):8.2f}  "
              f"{_m(r['got'] for r in rs):8.2f}  {_m(r['final_leak'] for r in rs):10.2f}  {_m(r['inter_leak'] for r in rs):10.2f}")
    print(f"all   {len(rows):>3}  {_m(r['exact'] for r in rows):.2f}  {'':8}  {'':8}  "
          f"{_m(r['final_leak'] for r in rows):10.2f}  {_m(r['inter_leak'] for r in rows):10.2f}")


# Pull the predicted final answer out of a plain-solve reply. Prefer the last "#### <n>" (the format
# SOLVE_PROMPT asks for); if the model didn't follow the format, fall back to the last number in the text,
# so a format slip isn't automatically marked wrong. Commas are removed first so "1,430" stays one number.
def extract_answer(text):
    t = text.replace(",", "")
    m = re.findall(r"####\s*\$?\s*(-?\d+(?:\.\d+)?)", t) or re.findall(r"-?\d+(?:\.\d+)?", t)
    return _norm(m[-1]) if m else None


# ---------- model side ----------
# The full tutoring rules written out as a system prompt. Only arm (c) (prompted=True) gets this: it's the
# "just tell the base model the rules" baseline. Trained arms see only the [LEVEL: n] tag and have to have
# learned the behaviour. The floor((n-1)*(S-1)/4) here is the same as target_k.
LEVEL_SPEC = """You are a math tutor. The student's message starts with a tag [LEVEL: n] (n from 1 to 5), then a word problem. Reply with a hint of depth n. NEVER state the final answer, and NEVER state the result of a step you are not told to show.
Let S be the number of calculation steps in a full solution. First show exactly floor((n-1)*(S-1)/4) finished worked steps from the start, each with its result. Then hint at the NEXT step:
- Level 1: restate the question; ask what is known and unknown. No calculations.
- Level 2: also say which quantities are related and how, without numbers or results.
- Level 3: also name the operation needed (add, multiply, ...), without computing it.
- Level 4: also write the next step as an unevaluated expression and tell the student to work it out.
- Level 5: show all worked steps except the last, then say "Now do the last step yourself."
"""
# Plain solve prompt with no level tag: checks how much ordinary math ability fine-tuning cost.
# The fixed "#### <number>" ending gives extract_answer something reliable to parse.
SOLVE_PROMPT = "Solve this problem step by step. End with the final answer on its own line as '#### <number>'.\n\n{q}"


# Format messages with the model's own chat template, ending with the assistant turn opener so the model
# writes the reply. enable_thinking=False turns off Qwen's <think> reasoning block: replies are
# then direct, fit in the token budget, and can't leak the answer inside hidden reasoning.
def _chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate(model, tok, prompts, max_new):
    """Greedy, left-padded, length-sorted batches. Returns (texts, truncated) in input order."""
    import torch
    # `tok` may be a multimodal processor that wraps the tokenizer; use the inner tokenizer if so.
    tk = getattr(tok, "tokenizer", tok)
    # Decoder-only models need left padding when batched: new tokens then follow each prompt's real last
    # token, not a run of pad tokens. Some tokenizers have no pad token, so fall back to EOS.
    tk.padding_side = "left"
    tk.pad_token = tk.pad_token or tk.eos_token
    # Sort prompts by length so each batch holds similar lengths: less padding, faster runs, and batches that
    # are the same every run. Results are put back in input order below.
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    texts, trunc = [None] * len(prompts), [None] * len(prompts)
    # A reply is finished if it emits either the tokenizer's EOS or the chat end-of-turn token <|im_end|>.
    eos = {tk.eos_token_id, tk.convert_tokens_to_ids("<|im_end|>")}
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        # add_special_tokens=False: the chat template already added every special token it needs; adding a
        # BOS again would make the input differ from what the model was trained on.
        enc = tk([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        # Greedy decoding (do_sample=False) makes runs reproducible and comparable across arms. The sampling
        # knobs are set to None to override any defaults in the model's generation_config, which would
        # otherwise raise warnings or leak into the call.
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                                 top_k=None, pad_token_id=tk.pad_token_id)
        for j, i in enumerate(idx):
            # With left padding every prompt in the batch ends at the same column, so slicing at the padded
            # input length leaves only the generated tokens.
            ids = out[j, enc["input_ids"].shape[1]:].tolist()
            # Truncated = never emitted a stop token, i.e. hit max_new. A cut-off reply can hide a leak that
            # would have come later, so the truncation rate is reported next to the scores.
            trunc[i] = not (eos & set(ids))
            texts[i] = tk.decode(ids, skip_special_tokens=True).strip()
    return texts, trunc


# Full evaluation of one arm: (1) tutor replies on every held problem at every level -> leakage/adherence
# table, (2) plain solve accuracy on the test slice, (3) everything, including raw replies, saved to
# <out_dir>/<name>.json so results can be checked or re-scored later without re-generating.
def run(model, tok, name, held, solve, prompted=False, out_dir="."):
    """prompted=True: arm (c), adds LEVEL_SPEC as a system prompt. Trained arms: user turn only."""
    # Each problem is asked at all 5 levels, so the same questions appear at every level and differences
    # between levels come from the level, not from which problems were asked.
    items = [(r, L) for r in held for L in LEVELS]
    prompts = []
    for r, L in items:
        # The user turn has the same "[LEVEL: n]\n<question>" layout as the training data.
        msgs = [{"role": "user", "content": f"[LEVEL: {L}]\n{r['question']}"}]
        prompts.append(_chat(tok, ([{"role": "system", "content": LEVEL_SPEC}] if prompted else []) + msgs))
    texts, trunc = generate(model, tok, prompts, TUTOR_MAX_NEW)
    rows = [{**score(t, r, L), "reply": t, "truncated": tr} for (r, L), t, tr in zip(items, texts, trunc)]
    summarize(rows, f"{name}: tutor ({len(held)} problems x 5 levels), truncated={_m(r['truncated'] for r in rows):.2f}")

    # Plain solve: no system prompt and no level tag, for every arm (prompted arms included), so the number
    # measures the model's own math ability and is comparable across arms.
    texts, trunc = generate(model, tok, [_chat(tok, [{"role": "user", "content": SOLVE_PROMPT.format(q=r["question"])}])
                                         for r in solve], SOLVE_MAX_NEW)
    srows = [{"pred": extract_answer(t), "gold": r["answer"], "reply": t, "truncated": tr}
             for r, t, tr in zip(solve, texts, trunc)]
    # Both sides went through _norm, so string equality is numeric equality ("18.00" == "18").
    acc = _m(s["pred"] == s["gold"] for s in srows)
    print(f"\n=== {name}: plain solve (GSM8K test[:{len(srows)}]) acc={acc:.3f} truncated={_m(s['truncated'] for s in srows):.2f}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"name": name, "prompted": prompted, "tutor": rows, "solve": srows, "solve_acc": acc},
              open(f"{out_dir}/{name}.json", "w"), ensure_ascii=False, indent=1)
    return rows, acc


# One line per arm from the JSON files that run() saved, so arms can be compared without re-generating.
# exact/leaks are over all held problems x 5 levels; solve_acc is the plain-solve "cost" number.
def compare(names, out_dir="."):
    print("arm              exact  final_leak  inter_leak  solve_acc  tutor_trunc")
    for n in names:
        d = json.load(open(f"{out_dir}/{n}.json"))
        t = d["tutor"]
        print(f"{n:<16} {_m(r['exact'] for r in t):5.2f}  {_m(r['final_leak'] for r in t):10.2f}  "
              f"{_m(r['inter_leak'] for r in t):10.2f}  {d['solve_acc']:9.3f}  {_m(r['truncated'] for r in t):11.2f}")


# ---------- selftest: harness must separate a perfect tutor from a leaky one ----------
# Two records in the notebook's parsed format, written out here so the selftest needs no download or model.
# Toula: 4 steps, every value distinct from the question, so all steps are scoreable.
# Billy: 3 steps, and step 1's value (3) also appears in the question, so it is not scoreable, which tests
# the ambiguity rule in score().
def _P(question, steps, final):
    return {"question": question, "final": final,
            "steps": [{"text": t, "expr": e, "val": v} for t, e, v in steps]}


_EX = [
    _P("Toula bought 3 dozen donuts at $68 per dozen, 2 dozen mini cupcakes at $80 per dozen, and 6 dozen mini "
       "cheesecakes at $55 per dozen. How much was the total cost?",
       [("3 x $68 = $204.", "3*68", "204"), ("2 x $80 = $160.", "2*80", "160"), ("6 x $55 = $330.", "6*55", "330"),
        ("$204 + $160 + $330 = $694.", "204+160+330", "694")], "694"),
    _P("Billy has 8 customers. His first 3 buy one DVD each. His next 2 buy 2 DVDs each. His last 3 buy none. "
       "How many DVDs did he sell?",
       [("3 * 1 = 3 DVDs.", "3*1", "3"), ("2 * 2 = 4 DVDs.", "2*2", "4"), ("3 + 4 + 0 = 7 DVDs.", "3+4+0", "7")], "7"),
]


# Checks the harness before trusting it on a real model: if the scorer can't tell a perfect tutor from one
# that dumps the whole solution, its numbers mean nothing. Runs in seconds with no GPU or data.
def selftest():
    # Unit checks: record conversion, the k formula, label stripping, answer extraction.
    recs = [to_rec(e) for e in _EX]
    assert recs[0]["steps"] == ["204", "160", "330", "694"] and recs[1]["answer"] == "7"
    assert [target_k(L, 4) for L in LEVELS] == [0, 0, 1, 2, 3]
    assert nums("Step 4: x.\n3. y [LEVEL: 2]", reply=True) == set()           # labels are not values
    assert extract_answer("so #### 1,430") == "1430" and extract_answer("it is $18.00.") == "18"
    # Two fake tutors scored at every level:
    #   oracle - states exactly the first k step results, then stops (the ideal reply)
    #   dump   - states every step result and the final answer (the worst reply)
    oracle, dump = [], []
    for r in recs:
        for L in LEVELS:
            k = target_k(L, len(r["steps"]))
            oracle.append(score("; ".join(f"result {v}" for v in r["steps"][:k]) + " Now you try.", r, L))
            dump.append(score(" ".join(f"result {v}" for v in r["steps"]) + f" answer {r['answer']}", r, L))
    summarize(oracle, "selftest oracle (want exact=1, leaks=0)")
    summarize(dump, "selftest full dump (want exact~0, leaks~1)")
    # The oracle must score perfectly. The dump must leak every time and rarely count as exact (only when
    # all scoreable steps fall inside k, which can happen for short problems at high levels).
    assert _m(s["exact"] for s in oracle) == 1 and _m(s["final_leak"] or s["inter_leak"] for s in oracle) == 0
    assert _m(s["final_leak"] for s in dump) == 1 and _m(s["exact"] for s in dump) < 0.3
    # Label vs value edge case: "Step 4" must not count as revealing Billy's step value 4, but stating 4 as
    # an actual quantity must.
    assert not score("Step 4: what is known?", recs[1], 1)["inter_leak"]       # 'Step 4' is not the value 4
    assert score("He sells 4 DVDs next.", recs[1], 1)["inter_leak"]
    print("selftest OK")


# Running the file directly just runs the selftest.
if __name__ == "__main__":
    selftest()
