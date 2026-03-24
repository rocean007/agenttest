#!/usr/bin/env python3
"""
AssetFlow Agent Tester — tests every LLM provider/model combination.

Usage:
  python test_agents.py                    # test all configured providers
  python test_agents.py --provider groq   # test only groq
  python test_agents.py --fast            # just ping test (no full analysis)
  python test_agents.py --symbol AAPL     # use specific test symbol
  python test_agents.py --workers 5       # parallel workers (default 3)
  python test_agents.py --save results.json  # save results to file

You need to fill in API keys below for the providers you want to test.
Free providers (no CC required): groq, openrouter, google, mistral, cerebras, ollama
"""

import os
import sys
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# FILL IN YOUR API KEYS HERE
# Get free keys from the signup URLs listed in the agents.json file
# ─────────────────────────────────────────────────────────────────────────────
API_KEYS = {
    "groq":        os.getenv("GROQ_API_KEY",        ""),  # console.groq.com
    "openrouter":  os.getenv("OPENROUTER_API_KEY",  ""),  # openrouter.ai
    "google":      os.getenv("GOOGLE_API_KEY",      ""),  # aistudio.google.com
    "mistral":     os.getenv("MISTRAL_API_KEY",     ""),  # console.mistral.ai
    "cerebras":    os.getenv("CEREBRAS_API_KEY",    ""),  # inference.cerebras.ai
    "together":    os.getenv("TOGETHER_API_KEY",    ""),  # platform.together.ai
    "deepseek":    os.getenv("DEEPSEEK_API_KEY",    ""),  # platform.deepseek.com
    "sambanova":   os.getenv("SAMBANOVA_API_KEY",   ""),  # cloud.sambanova.ai
    "nvidia":      os.getenv("NVIDIA_API_KEY",      ""),  # build.nvidia.com
    "cohere":      os.getenv("COHERE_API_KEY",      ""),  # cohere.com
    "fireworks":   os.getenv("FIREWORKS_API_KEY",   ""),  # fireworks.ai
    "hyperbolic":  os.getenv("HYPERBOLIC_API_KEY",  ""),  # app.hyperbolic.xyz
    "cloudflare":  os.getenv("CLOUDFLARE_API_KEY",  ""),  # dash.cloudflare.com
    "github":      os.getenv("GITHUB_TOKEN",        ""),  # github.com settings
    "huggingface": os.getenv("HF_TOKEN",            ""),  # huggingface.co
    "openai":      os.getenv("OPENAI_API_KEY",      ""),  # platform.openai.com
    "anthropic":   os.getenv("ANTHROPIC_API_KEY",   ""),  # console.anthropic.com
    "ollama":      "http://localhost:11434",               # local, no key needed
}

# Cloudflare also needs account ID
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

# ─────────────────────────────────────────────────────────────────────────────

TIMEOUT = 30
PING_PROMPT = "Reply with exactly the word: PONG"
ANALYSIS_PROMPT = """You are a financial analyst. Asset: AAPL (Apple Inc), equity.

Data context: Price $195.42 (+0.8% today), P/C ratio 0.95 (neutral), 
Analyst consensus: BUY, target $220, 52 analysts. Next earnings: 2 weeks.
Recent news: iPhone 16 sales strong, AI features driving upgrade cycle.

Output ONLY this JSON (fill in your real analysis):
{"signal":"bullish","confidence":72,"reasoning":"Strong iPhone cycle and AI differentiation support near-term upside","factors":["AI features","strong sales","positive analyst consensus"],"butterflies":["AI adoption -> upgrade cycle -> revenue beat -> analyst upgrades -> price target revision"]}"""

COLORS = {
    "GREEN": "\033[92m", "RED": "\033[91m", "YELLOW": "\033[93m",
    "BLUE": "\033[94m", "GRAY": "\033[90m", "BOLD": "\033[1m", "RESET": "\033[0m"
}

def c(color, text): return f"{COLORS.get(color,'')}{text}{COLORS['RESET']}"
def ok(msg): print(c("GREEN", f"  ✓ {msg}"))
def fail(msg): print(c("RED", f"  ✗ {msg}"))
def info(msg): print(c("BLUE", f"  → {msg}"))
def warn(msg): print(c("YELLOW", f"  ⚠ {msg}"))


def call_openai_compat(base_url, api_key, model, prompt, extra_headers=None):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if extra_headers: headers.update(extra_headers)
    r = requests.post(f"{base_url}/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 300, "temperature": 0.1},
        headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_google(api_key, model, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 300}}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_anthropic(api_key, model, prompt):
    r = requests.post("https://api.anthropic.com/v1/messages",
        json={"model": model, "max_tokens": 300,
              "messages": [{"role": "user", "content": prompt}]},
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def call_cohere(api_key, model, prompt):
    r = requests.post("https://api.cohere.com/v2/chat",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 300},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"][0]["text"]


def call_cloudflare(api_key, account_id, model, prompt):
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions"
    r = requests.post(url,
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 300},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["result"]["response"]


def call_huggingface(api_key, model, prompt):
    url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
    r = requests.post(url,
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 300},
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_ollama(base_url, model, prompt):
    r = requests.post(f"{base_url}/api/chat",
        json={"model": model, "stream": False,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60)
    r.raise_for_status()
    return r.json()["message"]["content"]


def test_agent(agent, test_type="ping"):
    provider = agent["provider"]
    model = agent["model"]
    api_key = API_KEYS.get(provider, "")

    # Skip if no key and key required
    if agent.get("requires_key") and provider not in ("ollama",) and not api_key:
        return {
            "status": "SKIP",
            "reason": f"No API key for {provider}",
            "time_ms": 0,
            "response": None,
        }

    prompt = PING_PROMPT if test_type == "ping" else ANALYSIS_PROMPT
    t0 = time.time()

    try:
        if provider == "google":
            resp = call_google(api_key, model, prompt)
        elif provider == "anthropic":
            resp = call_anthropic(api_key, model, prompt)
        elif provider == "cohere":
            resp = call_cohere(api_key, model, prompt)
        elif provider == "cloudflare":
            if not CLOUDFLARE_ACCOUNT_ID:
                return {"status": "SKIP", "reason": "No Cloudflare Account ID", "time_ms": 0}
            resp = call_cloudflare(api_key, CLOUDFLARE_ACCOUNT_ID, model, prompt)
        elif provider == "huggingface":
            resp = call_huggingface(api_key, model, prompt)
        elif provider == "ollama":
            resp = call_ollama(api_key, model, prompt)  # api_key is the base_url for ollama
        elif provider in ("groq", "openrouter", "mistral", "cerebras", "together",
                          "deepseek", "sambanova", "nvidia", "fireworks", "hyperbolic",
                          "openai", "github"):
            base_url = {
                "groq":       "https://api.groq.com/openai/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "mistral":    "https://api.mistral.ai/v1",
                "cerebras":   "https://api.cerebras.ai/v1",
                "together":   "https://api.together.xyz/v1",
                "deepseek":   "https://api.deepseek.com",
                "sambanova":  "https://api.sambanova.ai/v1",
                "nvidia":     "https://integrate.api.nvidia.com/v1",
                "fireworks":  "https://api.fireworks.ai/inference/v1",
                "hyperbolic": "https://api.hyperbolic.xyz/v1",
                "openai":     "https://api.openai.com/v1",
                "github":     "https://models.inference.ai.azure.com",
            }[provider]
            extra = {}
            if provider == "openrouter":
                extra = {"HTTP-Referer": "https://assetflow.local", "X-Title": "AssetFlow"}
            resp = call_openai_compat(base_url, api_key, model, prompt, extra)
        else:
            return {"status": "SKIP", "reason": f"Unknown provider: {provider}", "time_ms": 0}

        elapsed = int((time.time() - t0) * 1000)
        is_pong = "PONG" in resp.upper() if test_type == "ping" else bool(resp and len(resp) > 20)
        return {
            "status": "OK" if is_pong else "WARN",
            "reason": None if is_pong else "Response did not contain PONG",
            "time_ms": elapsed,
            "response": resp[:200] if resp else None,
        }

    except requests.exceptions.Timeout:
        return {"status": "FAIL", "reason": "Timeout", "time_ms": int((time.time()-t0)*1000)}
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        body = e.response.text[:200] if e.response else ""
        return {"status": "FAIL", "reason": f"HTTP {code}: {body}", "time_ms": int((time.time()-t0)*1000)}
    except Exception as e:
        return {"status": "FAIL", "reason": str(e)[:150], "time_ms": int((time.time()-t0)*1000)}


def print_result(agent, result):
    s = result["status"]
    t = result["time_ms"]
    name = agent["name"]
    if s == "OK":
        ok(f"{name} [{t}ms]")
    elif s == "WARN":
        warn(f"{name} [{t}ms] — {result.get('reason','')}")
        if result.get("response"):
            print(c("GRAY", f"    Response: {result['response'][:100]}"))
    elif s == "SKIP":
        print(c("GRAY", f"  ○ {name} — SKIP: {result.get('reason','')}"))
    else:
        fail(f"{name} — {result.get('reason','')}")


def main():
    parser = argparse.ArgumentParser(description="AssetFlow Agent Tester")
    parser.add_argument("--provider", help="Test only this provider")
    parser.add_argument("--model", help="Test only this model (requires --provider)")
    parser.add_argument("--fast", action="store_true", help="Ping test only (faster)")
    parser.add_argument("--full", action="store_true", help="Full analysis test with JSON output")
    parser.add_argument("--workers", type=int, default=3, help="Parallel workers (default 3)")
    parser.add_argument("--save", help="Save results to JSON file")
    parser.add_argument("--symbol", default="AAPL", help="Test symbol (cosmetic only)")
    parser.add_argument("--list", action="store_true", help="Just list all agents without testing")
    args = parser.parse_args()

    # Load agent list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    agents_file = os.path.join(script_dir, "agents.json")
    if not os.path.exists(agents_file):
        print(f"agents.json not found at {agents_file}")
        sys.exit(1)

    with open(agents_file) as f:
        all_agents = json.load(f)

    # Filter
    agents = all_agents
    if args.provider:
        agents = [a for a in agents if a["provider"] == args.provider]
    if args.model:
        agents = [a for a in agents if a["model"] == args.model]

    if args.list:
        print(f"\n{'Provider':<16} {'Model':<55} {'Free':<6} Signup")
        print("─" * 110)
        for a in agents:
            free_str = "✓ FREE" if a["free"] else "PAID"
            key_str = "no key" if not a.get("requires_key") else a.get("signup","")
            print(f"{a['provider']:<16} {a['model']:<55} {free_str:<6} {key_str}")
        print(f"\nTotal: {len(agents)} agents")
        return

    test_type = "full" if args.full else "ping"
    print(f"\n{c('BOLD', 'AssetFlow Agent Tester')}")
    print(f"Testing {len(agents)} agents | Mode: {test_type} | Workers: {args.workers}")
    print(f"Symbol: {args.symbol} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check which keys are set
    print(c("BLUE", "API Keys configured:"))
    for prov, key in API_KEYS.items():
        if prov == "ollama":
            print(f"  {prov:<16} {c('GREEN', '✓ local')}")
        elif key:
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            print(f"  {prov:<16} {c('GREEN', f'✓ {masked}')}")
        else:
            print(f"  {prov:<16} {c('GRAY', '○ not set')}")
    print()

    # Group by provider
    by_provider = {}
    for a in agents:
        by_provider.setdefault(a["provider"], []).append(a)

    results = []
    stats = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}

    for provider, pagents in by_provider.items():
        key_set = bool(API_KEYS.get(provider)) if provider != "ollama" else True
        print(c("BOLD", f"\n── {provider.upper()} ({len(pagents)} models) {'[NO KEY - SKIP]' if not key_set else ''}"))

        if args.workers > 1 and len(pagents) > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                fut_map = {ex.submit(test_agent, a, test_type): a for a in pagents}
                for fut in as_completed(fut_map):
                    agent = fut_map[fut]
                    result = fut.result()
                    result["agent"] = agent
                    results.append(result)
                    print_result(agent, result)
                    stats[result["status"]] = stats.get(result["status"], 0) + 1
        else:
            for a in pagents:
                result = test_agent(a, test_type)
                result["agent"] = a
                results.append(result)
                print_result(a, result)
                stats[result["status"]] = stats.get(result["status"], 0) + 1

    # Summary
    total = len(results)
    print(f"\n{'─'*60}")
    print(c("BOLD", "RESULTS SUMMARY"))
    print(f"  Total tested:  {total}")
    ok_n   = stats.get('OK',0)
    warn_n = stats.get('WARN',0)
    fail_n = stats.get('FAIL',0)
    skip_n = stats.get('SKIP',0)
    print('  ' + c('GREEN',  f'OK:    {ok_n}'))
    print('  ' + c('YELLOW', f'WARN:  {warn_n}'))
    print('  ' + c('RED',    f'FAIL:  {fail_n}'))
    print('  ' + c('GRAY',   f'SKIP:  {skip_n}'))

    working = [r for r in results if r["status"] in ("OK", "WARN")]
    if working:
        print(f"\n{c('GREEN', 'WORKING AGENTS:')}")
        for r in sorted(working, key=lambda x: x["time_ms"]):
            a = r["agent"]
            print(f"  {c('GREEN','✓')} {a['provider']:14} {a['model'][:50]:<50} {r['time_ms']:>6}ms")

    failed = [r for r in results if r["status"] == "FAIL"]
    if failed:
        print(f"\n{c('RED', 'FAILED AGENTS:')}")
        for r in failed:
            a = r["agent"]
            print(f"  {c('RED','✗')} {a['provider']:14} {a['model'][:40]:<40} {r.get('reason','')[:60]}")

    # Save results
    if args.save:
        output = {
            "tested_at": datetime.now().isoformat(),
            "test_type": test_type,
            "total": total,
            "stats": stats,
            "working": [{"provider":r["agent"]["provider"],"model":r["agent"]["model"],
                          "name":r["agent"]["name"],"time_ms":r["time_ms"]} for r in working],
            "failed": [{"provider":r["agent"]["provider"],"model":r["agent"]["model"],
                         "reason":r.get("reason","")} for r in failed],
            "all_results": [{**r, "agent": {"provider":r["agent"]["provider"],
                               "model":r["agent"]["model"],"name":r["agent"]["name"]}} for r in results],
        }
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n{c('BLUE', f'Results saved to {args.save}')}")

    print()


if __name__ == "__main__":
    main()
