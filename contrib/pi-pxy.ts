// installed by pxy
//
// `pxy launch pi` writes this file and rewrites it on every launch, so an edit
// here is an edit that goes; the source is contrib/pi-pxy.ts in the pxy repo.
// It registers pxy as a pi provider at startup, which is what the old
// ~/.pi/agent/models.json merge did — except a plain `pi`, started by hand,
// now gets the same catalog, and nothing on disk outside this file is touched.
//
// The catalog comes from `pxy models --json`, not from the daemon's
// /v1/models: that listing carries the `claude/`-prefixed mirror rows Claude
// Code's picker needs, which here would only list every model twice. Reading
// the CLI also means a pi started while the daemon is down still knows the
// models it will have once the daemon is back.
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const run = promisify(execFile);

// Substituted by `pxy launch pi` when it writes this file: the daemon's v1
// endpoint as config.toml spells it, and the absolute path of the pxy that did
// the writing — pi's PATH is not necessarily the shell's.
const BASE_URL = "__PXY_BASE_URL__";
const PXY_BIN = "__PXY_BIN__";

// pi waits for this factory before startup continues, so a wedged pxy would be
// a pi that never draws. Read the catalog under a deadline and go without.
const CATALOG_TIMEOUT_MS = 5000;

// One row of `pxy models --json`: a group (a named failover chain, whose name
// is itself a model id) or a single provider/model.
type CatalogRow = {
  id: string;
  kind: "group" | "model";
  label?: string;
  name?: string | null;
  contextLength: number;
  maxOutputTokens: number;
  reasoning?: boolean | null;
};

async function catalog(): Promise<ProviderModelConfig[]> {
  const { stdout } = await run(PXY_BIN, ["models", "--json"], {
    timeout: CATALOG_TIMEOUT_MS,
    maxBuffer: 16 * 1024 * 1024,
  });
  return (JSON.parse(stdout) as CatalogRow[]).map((row) => ({
    id: row.id,
    // Groups carry a label ("Plans"); a model row's name is whatever
    // config.toml gave it, which is usually nothing.
    name: row.label ?? row.name ?? row.id,
    // Asserted per model in config.toml. Unset reads as no, and a group only
    // reasons when every member of its chain does — a thinking level pi offers
    // for a model that has none is a capability it shows and cannot deliver.
    reasoning: row.reasoning === true,
    // pxy's config has no per-model image capability to report, so every row
    // is text — the same thing the models.json merge advertised.
    input: ["text"],
    // Zero across the board: what a turn costs depends on which member of a
    // chain served it, which pi cannot know and `pxy status` already tracks. A
    // plausible-looking number here would only be a wrong one.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: row.contextLength,
    maxTokens: row.maxOutputTokens,
  }));
}

export default async function (pi: ExtensionAPI) {
  let models: ProviderModelConfig[] = [];
  let failure: string | undefined;
  try {
    models = await catalog();
  } catch (err) {
    failure = err instanceof Error ? err.message : String(err);
  }

  pi.registerProvider("pxy", {
    name: "pxy",
    baseUrl: BASE_URL,
    // The key is a soft gate pxy never validates (loopback only). What it does
    // read is which agent is asking, for the per-model usage stats, and
    // `x-pxy-agent` says that outright — the other agents have to smuggle it
    // as a `:pi` suffix on the key because they cannot be taught a header.
    apiKey: "pxy",
    headers: { "x-pxy-agent": "pi" },
    api: "openai-completions",
    models,
    // config.toml is the whole catalog and it changes by hand. This is how a
    // model added there arrives without restarting pi.
    refreshModels: async () => await catalog(),
  });

  // Nothing can be said from the factory — there is no UI yet — and a provider
  // that quietly registered zero models reads as pi's fault rather than pxy's.
  if (failure) {
    pi.on("session_start", (_event, ctx) => {
      ctx.ui.notify(`pxy: no models — \`${PXY_BIN} models --json\` failed: ${failure}`, "error");
    });
  }
}
