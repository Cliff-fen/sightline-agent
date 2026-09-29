import {
  createModels,
  createProvider,
  envApiKeyAuth,
  type Api,
  type Model,
  type Models,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import { anthropicProvider } from "@earendil-works/pi-ai/providers/anthropic";

export type ModelProfileName = "local" | "claude" | "deepseek" | "openai";

type Profile = {
  provider: string;
  modelId: string;
  apiKeyEnv?: string;
  defaultBaseUrl: string;
};

export type RuntimeConfig = {
  profile: ModelProfileName;
  models: Models;
  model: Model<Api>;
  apiKeyEnv?: string;
  configured: boolean;
  toolGatewayUrl: string;
  maxTurns: number;
  maxImageBytes: number;
};

export const MODEL_PROFILES: Readonly<Record<ModelProfileName, Profile>> = {
  local: {
    provider: "sightline-local",
    modelId: "local-model",
    defaultBaseUrl: "http://127.0.0.1:8000/v1",
  },
  claude: {
    provider: "anthropic",
    modelId: "claude-sonnet-5",
    apiKeyEnv: "ANTHROPIC_API_KEY",
    defaultBaseUrl: "https://api.anthropic.com",
  },
  deepseek: {
    provider: "deepseek",
    modelId: "deepseek-chat",
    apiKeyEnv: "DEEPSEEK_API_KEY",
    defaultBaseUrl: "https://api.deepseek.com",
  },
  openai: {
    provider: "openai",
    modelId: "gpt-4.1",
    apiKeyEnv: "OPENAI_API_KEY",
    defaultBaseUrl: "https://api.openai.com/v1",
  },
};

function positiveInteger(value: string | undefined, fallback: number, name: string): number {
  if (value === undefined || value === "") return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`${name} must be a positive integer`);
  return parsed;
}

function modelInput(env: NodeJS.ProcessEnv, fallback: ("text" | "image")[]): ("text" | "image")[] {
  if (!env.MODEL_INPUT) return fallback;
  const input = env.MODEL_INPUT.split(",").map((value) => value.trim()).filter(Boolean);
  if (input.length === 0 || input.some((value) => value !== "text" && value !== "image")) {
    throw new Error("MODEL_INPUT must be a comma-separated subset of: text,image");
  }
  return [...new Set(input)] as ("text" | "image")[];
}

function httpUrl(value: string, name: string): string {
  let url: URL;
  try { url = new URL(value); }
  catch { throw new Error(`${name} must be a valid URL`); }
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error(`${name} must use http or https`);
  return value.replace(/\/$/, "");
}

function profileName(value: string): ModelProfileName {
  if (!(value in MODEL_PROFILES)) {
    throw new Error(`MODEL_PROFILE must be one of: ${Object.keys(MODEL_PROFILES).join(", ")} (received ${value})`);
  }
  return value as ModelProfileName;
}

function localModel(id: string, baseUrl: string, env: NodeJS.ProcessEnv): Model<"openai-completions"> {
  return {
    id,
    name: id,
    api: "openai-completions",
    provider: "sightline-local",
    baseUrl,
    reasoning: false,
    input: modelInput(env, ["text", "image"]),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: positiveInteger(env.MODEL_CONTEXT_WINDOW, 65_536, "MODEL_CONTEXT_WINDOW"),
    maxTokens: positiveInteger(env.MODEL_MAX_TOKENS, 8_192, "MODEL_MAX_TOKENS"),
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      maxTokensField: "max_tokens",
    },
  };
}

function deepSeekModel(id: string, baseUrl: string, env: NodeJS.ProcessEnv): Model<"openai-completions"> {
  return {
    id,
    name: id,
    api: "openai-completions",
    provider: "deepseek",
    baseUrl,
    reasoning: false,
    input: modelInput(env, ["text"]),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: positiveInteger(env.MODEL_CONTEXT_WINDOW, 1_000_000, "MODEL_CONTEXT_WINDOW"),
    maxTokens: positiveInteger(env.MODEL_MAX_TOKENS, 128_000, "MODEL_MAX_TOKENS"),
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      maxTokensField: "max_tokens",
      thinkingFormat: "deepseek",
    },
  };
}

function relayOpenAIModel(id: string, baseUrl: string, env: NodeJS.ProcessEnv): Model<"openai-responses"> {
  return {
    id,
    name: id,
    api: "openai-responses",
    provider: "openai-relay",
    baseUrl,
    reasoning: false,
    input: modelInput(env, ["text", "image"]),
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: positiveInteger(env.MODEL_CONTEXT_WINDOW, 128_000, "MODEL_CONTEXT_WINDOW"),
    maxTokens: positiveInteger(env.MODEL_MAX_TOKENS, 16_384, "MODEL_MAX_TOKENS"),
    compat: {
      supportsDeveloperRole: false,
      supportsStrictMode: true,
    },
  };
}

export function loadRuntimeConfig(env: NodeJS.ProcessEnv = process.env): RuntimeConfig {
  const profile = profileName(env.MODEL_PROFILE ?? "local");
  const spec = MODEL_PROFILES[profile];
  const modelId = env.MODEL_NAME || spec.modelId;
  const baseUrl = httpUrl(env.MODEL_BASE_URL ?? spec.defaultBaseUrl, "MODEL_BASE_URL");
  const models = createModels();

  if (profile === "claude") {
    models.setProvider(anthropicProvider());
  } else if (profile === "openai") {
    const model = relayOpenAIModel(modelId, baseUrl, env);
    models.setProvider(createProvider({
      id: "openai-relay",
      name: "OpenAI-compatible relay",
      baseUrl,
      auth: { apiKey: envApiKeyAuth("OpenAI API key", ["OPENAI_API_KEY"]) },
      models: [model],
      api: openAIResponsesApi(),
    }));
  } else if (profile === "deepseek") {
    const model = deepSeekModel(modelId, baseUrl, env);
    models.setProvider(createProvider({
      id: "deepseek",
      name: "DeepSeek",
      baseUrl,
      auth: { apiKey: envApiKeyAuth("DeepSeek API key", ["DEEPSEEK_API_KEY", "MODEL_API_KEY"]) },
      models: [model],
      api: openAICompletionsApi(),
    }));
  } else {
    const model = localModel(modelId, baseUrl, env);
    models.setProvider(createProvider({
      id: "sightline-local",
      name: "Sightline local vLLM",
      baseUrl,
      auth: {
        apiKey: {
          name: "Local model key",
          resolve: async () => ({ auth: { apiKey: env.MODEL_API_KEY || "local" }, source: "local runtime" }),
        },
      },
      models: [model],
      api: openAICompletionsApi(),
    }));
  }

  const providerId = profile === "openai" ? "openai-relay" : spec.provider;
  const selectedModel = models.getModel(providerId, modelId);
  const defaultModel = selectedModel ?? models.getModel(providerId, spec.modelId);
  if (!defaultModel) throw new Error(`Pi provider ${spec.provider} does not include ${spec.modelId}`);
  const catalogModel = selectedModel ?? {
    ...defaultModel,
    id: modelId,
    name: modelId,
  };
  const model = baseUrl === catalogModel.baseUrl ? catalogModel : { ...catalogModel, baseUrl };
  const apiKeyEnv = spec.apiKeyEnv;
  const configured = !apiKeyEnv || Boolean(env[apiKeyEnv]);

  return {
    profile,
    models,
    model,
    ...(apiKeyEnv ? { apiKeyEnv } : {}),
    configured,
    toolGatewayUrl: httpUrl(env.TOOL_GATEWAY_URL ?? "http://127.0.0.1:8090", "TOOL_GATEWAY_URL"),
    maxTurns: positiveInteger(env.MAX_TURNS, 20, "MAX_TURNS"),
    maxImageBytes: positiveInteger(env.MODEL_MAX_IMAGE_BYTES, 20 * 1024 * 1024, "MODEL_MAX_IMAGE_BYTES"),
  };
}
