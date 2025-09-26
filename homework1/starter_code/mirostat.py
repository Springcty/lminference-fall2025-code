import os
import math
import time

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1, plot_logits_dist=False):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) # (1, 6)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"

    # TODO: YOUR CODE HERE -- additional variable init
    # We will not be checking this section for correctness,
    # But you will probably eventually want to set up some 
    # extra variables here for plotting metrics.
    # Our advice is to fill out the other sections first!
    ks, s_hats, mus, errors, surprisals = [], [], [], [], []
    
    # for logits distribution plotting
    step_logits = {}

    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)
        
        # TODO: YOUR CODE HERE -- Estimate Zipf's exponent
        # Following Basu et al, use m=100 (i.e. use only the top 100 tokens(' diffs) to estimate the exponent)
        # Refer to Equation 30 https://arxiv.org/pdf/2007.14966#equation.C.30 for pointers
        m = 100
        l = sorted_logits[0, :m]
        t = torch.log((torch.arange(2, m + 1, device=device, dtype=torch.float) / torch.arange(1, m, device=device, dtype=torch.float)))
        b = l[:-1] - l[1:] # b = log p(i) - log p(i+1) = logit_i - logit_{i+1}
        s_hat = (t @ b) / (t @ t)
        s_hat_value = s_hat.item()

        # TODO: YOUR CODE HERE -- Compute k using Zipf exponent
        N = adjusted_probs.size(1)
        eps = s_hat_value - 1.0
        k = ((eps * math.exp(mu)) / (1 - float(N) ** (-eps))) ** (1.0 / s_hat_value)
        k = int(round(k))
        k = max(1, min(k, N))
        # print('*' * 20, k, '*' * 20)

        # capture logits distribution at selected steps {1, 10, 100}
        if step in {1, 10, 100} and step not in step_logits:
            sl = sorted_logits[0].detach().float().cpu().numpy()
            step_logits[step] = {"sorted_logits": sl, "k": k}
        
        # top k sampling
        topk_logits = sorted_logits[:, 0:k]
        topk_inds = sorted_inds[:, 0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

        # TODO: YOUR CODE HERE -- Compute surprisal error and adjust mu accordingly
        prob_next = topk_probs[0, (topk_inds == next_tok).nonzero(as_tuple=True)[1]]
        surprisal = -torch.log(prob_next).item()

        err = surprisal - target_ce
        mu = mu - learning_rate * err

        # Record metrics for plotting
        ks.append(k)
        s_hats.append(float(s_hat))
        mus.append(mu)
        errors.append(float(err))
        surprisals.append(surprisal)

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    per_tok_ppl = torch.exp(torch.tensor(surprisals)).cpu().numpy()
    seq_ppl = float(torch.exp(torch.mean(torch.tensor(surprisals))).cpu().numpy())
    stats = {
        'k': ks,
        's_hat': s_hats,
        'mu': mus,
        'error': errors,
        'surprisal': surprisals,
        'per_token_ppl': per_tok_ppl,
        'seq_ppl': seq_ppl,
        'generated_tokens': len(surprisals),
        'final_len': input_ids.size(1)
    }

    if plot_logits_dist:
        plt.figure(figsize=(9, 6))
        for step in sorted(step_logits.keys()):
            sl = step_logits[step]["sorted_logits"]
            k = step_logits[step]["k"]
            n = sl.size
            ranks = np.arange(1, n + 1)
            # one curve per step
            plt.plot(ranks, sl[:n], label=f"step {step} (k={k})")
            if 1 <= k <= n:
                plt.axvline(x=k, linestyle="--")

        plt.xlabel("Rank")
        plt.ylabel("Logit (temp-adjusted)")
        plt.title(f"Logit distributions (one plot) | τ={tau} | prompt='{prompt}' | {model_name}")
        plt.legend()
        plt.tight_layout()
        
        outfile = os.path.join(
            './mirostat_plots',
            f"logits_{tokenizer.name_or_path.split('/')[-1]}_tau{target_ce}.png",
        )
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"Saved: {outfile}")
    return text, stats

def plot_run(stats, title, outfile):
    """Single-axis figure with k, s_hat, mu, error curves."""
    dirpath, filename = os.path.split(outfile)
    basename, ext = os.path.splitext(filename)
    outfile_1 = f"{basename}_s-hat_mu_error{ext}"
    outfile_1 = os.path.join(dirpath, outfile_1)
    outfile_2 = f"{basename}_k{ext}"
    outfile_2 = os.path.join(dirpath, outfile_2)
    
    steps = np.arange(1, len(stats["k"]) + 1)
    plt.figure(figsize=(9, 6))
    # plt.plot(steps, stats["k"], label="k")
    plt.plot(steps, stats["s_hat"], label="s_hat")
    plt.plot(steps, stats["mu"], label="mu")
    plt.plot(steps, stats["error"], label="surprisal error")
    plt.xlabel("Generation step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outfile_1, dpi=150)
    plt.close()

    steps = np.arange(1, len(stats["k"]) + 1)
    plt.figure(figsize=(9, 6))
    plt.plot(steps, stats["k"], label="k")
    plt.xlabel("Generation step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile_2, dpi=150)
    plt.close()

def print_report(prompt, model_name, tau, text, stats):
    ptp = stats["per_token_ppl"]
    mean_ppl = float(np.mean(ptp)) if len(ptp) else float("nan")
    median_ppl = float(np.median(ptp)) if len(ptp) else float("nan")
    std_ppl = float(np.std(ptp, ddof=0)) if len(ptp) else float("nan")
    seq_ppl = stats["seq_ppl"]

    print("=" * 100)
    print(f"Model: {model_name}")
    print(f"Prompt: {repr(prompt)}")
    print(f"tau (target CE): {tau}")
    print(f"Total length: {stats['final_len']} (generated {stats['generated_tokens']} tokens)")
    print("\nGenerated sequence:\n")
    print(text)
    print("\nPer-token perplexity stats:")
    print(f"  Mean   : {mean_ppl:.4f}")
    print(f"  Median : {median_ppl:.4f}")
    print(f"  Std    : {std_ppl:.4f}")
    print(f"Sequence-level perplexity: {seq_ppl:.4f}")


def load_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    if device == "cuda":
        model.to(device)
    return tokenizer, model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("mirostat_plots", exist_ok=True)

    taus = [2.0, 3.0, 4.0]               # three target cross-entropies
    prompts = ["Once upon a time,", "3 + 5 = "]  # two prompts
    models = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B"]  # two model sizes

    for model_name in models:
        tokenizer, model = load_model(model_name, device)
        for prompt in prompts:
            for tau in taus:
                t0 = time.time()
                text, stats = mirostat(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    target_ce=tau,
                    learning_rate=0.1,
                    temperature=0.9,
                    max_length=128,
                    device=device,
                )
                dt = time.time() - t0

                # report
                print_report(prompt, model_name, tau, text, stats)
                print(f"Runtime: {dt:.2f} s")

                # plot
                safe_prompt = prompt.replace(" ", "_").replace(",", "").replace("'", "")
                fig_name = f"mirostat_plots/{model_name.split('/')[-1]}__{safe_prompt}__tau{tau}.png"
                plot_title = f"{model_name} | prompt='{prompt}' | tau={tau}"
                plot_run(stats, plot_title, fig_name)
                print(f"Saved plot to: {fig_name}\n")

if __name__ == "__main__":
    # main()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("mirostat_plots", exist_ok=True)

    tau = 2.0
    prompt = "Once upon a time,"
    model_name = "meta-llama/Llama-3.2-1B"
    tokenizer, model = load_model(model_name, device)

    text, stats = mirostat(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        target_ce=tau,
        learning_rate=0.1,
        temperature=0.9,
        max_length=128,
        device=device,
        plot_logits_dist=True,
    )
    
# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-3.1-8B"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.eval()
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     prompt = "Once upon a time,"
#     result, stats = mirostat(model, tokenizer, prompt, max_length=256, device=device, temperature=1.0, target_ce=3.0, learning_rate=0.1)
#     print(result)