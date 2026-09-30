import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export type AgentContract = {
  version: number;
  maxTurns: number;
  toolTimeoutMs: number;
  systemPrompt: string;
  tools: string[];
};

function contractPath(): string {
  const here = dirname(fileURLToPath(import.meta.url));
  const candidates = [
    resolve(here, "../../shared/agent-contract.json"),
    resolve(here, "../../../shared/agent-contract.json"),
    resolve(process.cwd(), "shared/agent-contract.json"),
    resolve(process.cwd(), "../shared/agent-contract.json"),
  ];
  const path = candidates.find(existsSync);
  if (!path) throw new Error("Cannot locate shared/agent-contract.json");
  return path;
}

function loadAgentContract(): AgentContract {
  const value = JSON.parse(readFileSync(contractPath(), "utf8")) as Partial<AgentContract>;
  if (!Number.isSafeInteger(value.version) || !Number.isSafeInteger(value.maxTurns)
    || !Number.isSafeInteger(value.toolTimeoutMs) || typeof value.systemPrompt !== "string"
    || !Array.isArray(value.tools) || value.tools.some((tool) => typeof tool !== "string")) {
    throw new Error("shared/agent-contract.json is invalid");
  }
  return value as AgentContract;
}

export const AGENT_CONTRACT = Object.freeze(loadAgentContract());
